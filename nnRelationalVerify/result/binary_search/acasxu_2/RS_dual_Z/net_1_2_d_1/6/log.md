## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 5471.923754115261


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344)
1: (-1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438)
2: (-1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969)
3: (-1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930)
4: (-1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016)

## BASE Result
execution time: IAR + LP analysis = 1.89 + 2.39 = 4.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -5471.9793324, upper bound: 5471.9793324


# Binary Search by BASE starts (time budget: 1195.73 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=6162.86279296875
rel_dist={3: [-5471.979332446701, 5471.979332446701]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=6162.86279296875
rel_dist={3: [-5471.979020840545, 5471.979020840547]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=6162.86279296875
rel_dist={3: [-5471.978598948089, 5471.97859894809]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=6162.86279296875
rel_dist={3: [-5471.978217283832, 5471.978217283833]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=6162.86279296875
rel_dist={3: [-5471.977887691975, 5471.977887691801]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=6162.86279296875
rel_dist={3: [-5471.9776877308705, 5471.977687730634]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=6162.86279296875
rel_dist={3: [-5471.977546026561, 5471.977546026394]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=6162.86279296875
rel_dist={3: [-5471.977467799268, 5471.977467799357]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=6162.86279296875
rel_dist={3: [-5471.977426835224, 5471.977426835225]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=6162.86279296875
rel_dist={3: [-5471.977406353012, 5471.977406353071]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=6162.86279296875
rel_dist={3: [-5471.977396111982, 5471.977396112639]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=6162.86279296875
rel_dist={3: [-5471.977390991502, 5471.9773909915275]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=6162.86279296875
rel_dist={3: [-5471.977388431275, 5471.977388431009]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=6162.86279296875
rel_dist={3: [-5471.9773871511625, 5471.977387150775]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=6162.86279296875
rel_dist={3: [-5471.977386511208, 5471.977386511064]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=6162.86279296875
rel_dist={3: [-5471.977386191007, 5471.977386190945]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=6162.86279296875
rel_dist={3: [-5471.977386031454, 5471.977386031294]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=6162.86279296875
rel_dist={3: [-5471.977385951257, 5471.97738595117]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=6162.86279296875
rel_dist={3: [-5471.977385911711, 5471.977385912429]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=6162.86279296875
rel_dist={3: [-5471.977385894648, 5471.9773858932]}

## Binary Search Result
Binary search time: 87.37 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1108.35 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786145, upper bound: 5471.9786145
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786145, upper bound: 5471.9786145
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.21
Output dim: 3, lower bound: -5471.9786145, upper bound: 5471.9786145
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.21
Output dim: 3, lower bound: -5471.9786145, upper bound: 5471.9786145

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784161, upper bound: 5471.9784859
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784161
time: 0.95 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784161, upper bound: 5471.9784859
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784161
time: 0.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -5471.9784161, upper bound: 5471.9784859
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784161
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -5471.9784161, upper bound: 5471.9784859
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784161

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9783529
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9783473
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783473, upper bound: 5471.9782590
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9782406
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9783529
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9783473
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783473, upper bound: 5471.9782589
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9782406
time: 0.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9783529
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9783473
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9783473, upper bound: 5471.9782590
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9782406
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9783529
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9783473
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9783473, upper bound: 5471.9782589
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9782406

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9782282
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781590, upper bound: 5471.9783516
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9782335
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781602, upper bound: 5471.9783461
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783461, upper bound: 5471.9781602
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782335, upper bound: 5471.9782590
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9781590
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782289, upper bound: 5471.9782406
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9782289
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781590, upper bound: 5471.9783516
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9782335
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781602, upper bound: 5471.9783461
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783461, upper bound: 5471.9781602
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782335, upper bound: 5471.9782590
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9781590
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782282, upper bound: 5471.9782406
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9782282
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9781590, upper bound: 5471.9783516
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9782335
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9781602, upper bound: 5471.9783461
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9783461, upper bound: 5471.9781602
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782335, upper bound: 5471.9782590
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9781590
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782289, upper bound: 5471.9782406
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782406, upper bound: 5471.9782289
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9781590, upper bound: 5471.9783516
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782590, upper bound: 5471.9782335
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9781602, upper bound: 5471.9783461
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9783461, upper bound: 5471.9781602
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782335, upper bound: 5471.9782590
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9781590
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 3, lower bound: -5471.9782282, upper bound: 5471.9782406

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758782, upper bound: 5471.9758808
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758808
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759789
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759790
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758425
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758424
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758815, upper bound: 5471.9758783
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758815, upper bound: 5471.9758782
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758815
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758815
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759945
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759789
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759789
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758425
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758424
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758424
time: 2.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758808, upper bound: 5471.9758782
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758808, upper bound: 5471.9758783
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758782, upper bound: 5471.9758808
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758808
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759790
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758425
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758424
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758815, upper bound: 5471.9758783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758815, upper bound: 5471.9758782
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758815
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758783, upper bound: 5471.9758815
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759945
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759945
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759295, upper bound: 5471.9758878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758424, upper bound: 5471.9759789
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758425, upper bound: 5471.9759789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758425
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759790, upper bound: 5471.9758424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758878, upper bound: 5471.9759295
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758424
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9759945, upper bound: 5471.9758425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758808, upper bound: 5471.9758782
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 3, lower bound: -5471.9758808, upper bound: 5471.9758783

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757732
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757732
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757897
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757898
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757483
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9757483
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9758153
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9757483
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9758153
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757768
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757768
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757768
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757768
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757897
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757898
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757951, upper bound: 5471.9757483
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757951, upper bound: 5471.9757483
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9758661
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757732, upper bound: 5471.9758153
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757732, upper bound: 5471.9758153
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757732
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757732
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757897
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757898
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9758153
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9757483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757768, upper bound: 5471.9758153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758153, upper bound: 5471.9757768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757492, upper bound: 5471.9757962
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9758661, upper bound: 5471.9757869
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757588, upper bound: 5471.9757898
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757951
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9759221
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757951, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759221, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757951, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9758661
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757898, upper bound: 5471.9757588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757869, upper bound: 5471.9758661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9759335, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757962, upper bound: 5471.9757492
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757732, upper bound: 5471.9758153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757483, upper bound: 5471.9757483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.02
Output dim: 3, lower bound: -5471.9757732, upper bound: 5471.9758153

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746104, upper bound: 5471.9745890
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746104, upper bound: 5471.9745890
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746104, upper bound: 5471.9745890
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746104, upper bound: 5471.9745890
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9747658
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747658, upper bound: 5471.9745890
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9746777
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746777, upper bound: 5471.9745890
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746777, upper bound: 5471.9745890
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746777, upper bound: 5471.9745890
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746777, upper bound: 5471.9745890
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745890, upper bound: 5471.9745890
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=6162.86279296875
rel_dist={3: [-5471.979332446701, 5471.979332446701]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783526, upper bound: 5471.9783526
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783526, upper bound: 5471.9783526
time: 0.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 3, lower bound: -5471.9783526, upper bound: 5471.9783526
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 3, lower bound: -5471.9783526, upper bound: 5471.9783526

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9782387
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9781940
time: 1.08 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9782387
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782387, upper bound: 5471.9781940
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9782387
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9781940
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 3, lower bound: -5471.9781940, upper bound: 5471.9782387
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 3, lower bound: -5471.9782387, upper bound: 5471.9781940

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9781953
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781192, upper bound: 5471.9781837
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9781192
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780761
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9781953
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781192, upper bound: 5471.9781837
time: 8.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9781193
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9780761
time: 1.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9781953
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781192, upper bound: 5471.9781837
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9781192
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780761
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9781953
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781192, upper bound: 5471.9781837
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9781193
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.83
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9780761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9780487
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779936, upper bound: 5471.9781953
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780496
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780130, upper bound: 5471.9781837
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9780130
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9781193
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9779936
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780487, upper bound: 5471.9780761
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9780488
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779936, upper bound: 5471.9781953
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780496
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780130, upper bound: 5471.9781837
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9780130
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9781193
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9779936
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780488, upper bound: 5471.9780761
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9780487
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9779936, upper bound: 5471.9781953
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780496
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780130, upper bound: 5471.9781837
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9780130
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9781193
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9779936
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780487, upper bound: 5471.9780761
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780761, upper bound: 5471.9780488
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9779936, upper bound: 5471.9781953
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9781193, upper bound: 5471.9780496
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780130, upper bound: 5471.9781837
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9780130
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780496, upper bound: 5471.9781193
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9781953, upper bound: 5471.9779936
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9780488, upper bound: 5471.9780761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757927
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757927
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757965
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757966
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757602
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757966, upper bound: 5471.9758183
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757537
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757536
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9757798
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757932, upper bound: 5471.9757798
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757932
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757932
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757966
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757965
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757536
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757537
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757927, upper bound: 5471.9757798
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757927, upper bound: 5471.9757798
time: 1.00 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757927
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757965
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757966
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757602
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757966, upper bound: 5471.9758183
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757537
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757536
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9757798
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757932, upper bound: 5471.9757798
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757798, upper bound: 5471.9757932
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757537, upper bound: 5471.9758853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757966
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758183, upper bound: 5471.9757965
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757603, upper bound: 5471.9758787
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758787, upper bound: 5471.9757602
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757965, upper bound: 5471.9758183
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757536
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9758853, upper bound: 5471.9757537
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757927, upper bound: 5471.9757798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.20
Output dim: 3, lower bound: -5471.9757927, upper bound: 5471.9757798

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757210, upper bound: 5471.9756857
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756857
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756808
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756635
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757077, upper bound: 5471.9756635
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756846, upper bound: 5471.9756635
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756878, upper bound: 5471.9757206
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756846, upper bound: 5471.9756635
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756878, upper bound: 5471.9757210
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757210, upper bound: 5471.9756878
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756846
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757206, upper bound: 5471.9756878
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756846
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757077, upper bound: 5471.9756635
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9756635
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756857, upper bound: 5471.9757206
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756857, upper bound: 5471.9757210
time: 1.24 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757210, upper bound: 5471.9756857
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756857
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757077, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756846, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756878, upper bound: 5471.9757206
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756846, upper bound: 5471.9756635
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756878, upper bound: 5471.9757210
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757210, upper bound: 5471.9756878
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756846
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757206, upper bound: 5471.9756878
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9758194
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757577, upper bound: 5471.9756970
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756808, upper bound: 5471.9756949
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9757080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9758166
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758166, upper bound: 5471.9756637
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757080, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756949, upper bound: 5471.9756808
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756970, upper bound: 5471.9757577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9757077, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9758194, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756637, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756857, upper bound: 5471.9757206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756635, upper bound: 5471.9756635
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 3, lower bound: -5471.9756857, upper bound: 5471.9757210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745825, upper bound: 5471.9747279
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745824
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745824
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746058, upper bound: 5471.9745825
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746058, upper bound: 5471.9745825
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746058, upper bound: 5471.9745825
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746058, upper bound: 5471.9745825
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745825, upper bound: 5471.9747279
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9747279
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9747279, upper bound: 5471.9745825
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9746668
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746668, upper bound: 5471.9745825
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746668, upper bound: 5471.9745824
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745824
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9745824, upper bound: 5471.9745825
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746668, upper bound: 5471.9745825
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746668, upper bound: 5471.9745825
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=6162.86279296875
rel_dist={3: [-5471.979020840545, 5471.979020840547]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779187, upper bound: 5471.9779072
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779072, upper bound: 5471.9779187
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 3, lower bound: -5471.9779187, upper bound: 5471.9779072
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 3, lower bound: -5471.9779072, upper bound: 5471.9779187

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777203, upper bound: 5471.9778005
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777203, upper bound: 5471.9777019
time: 1.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777019, upper bound: 5471.9778182
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778005, upper bound: 5471.9777203
time: 1.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 3, lower bound: -5471.9777203, upper bound: 5471.9778005
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 3, lower bound: -5471.9777203, upper bound: 5471.9777019
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 3, lower bound: -5471.9777019, upper bound: 5471.9778182
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 3, lower bound: -5471.9778005, upper bound: 5471.9777203

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776994, upper bound: 5471.9777602
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9777945
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776994, upper bound: 5471.9776960
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778062, upper bound: 5471.9776988
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776988, upper bound: 5471.9778061
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776960, upper bound: 5471.9777969
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777945, upper bound: 5471.9777104
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777602, upper bound: 5471.9776994
time: 1.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9776994, upper bound: 5471.9777602
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9777945
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9776994, upper bound: 5471.9776960
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9778062, upper bound: 5471.9776988
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9776988, upper bound: 5471.9778061
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9776960, upper bound: 5471.9777969
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9777945, upper bound: 5471.9777104
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.14
Output dim: 3, lower bound: -5471.9777602, upper bound: 5471.9776994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9777014
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9777602
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9777229
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776423, upper bound: 5471.9777945
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777970, upper bound: 5471.9776410
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777229, upper bound: 5471.9776960
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778062, upper bound: 5471.9776444
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776423, upper bound: 5471.9776988
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776988, upper bound: 5471.9777192
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9778061
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776960, upper bound: 5471.9777229
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776410, upper bound: 5471.9777970
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777945, upper bound: 5471.9776423
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777229, upper bound: 5471.9777104
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777602, upper bound: 5471.9776444
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777014, upper bound: 5471.9776994
time: 1.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9777014
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9777602
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9777229
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776423, upper bound: 5471.9777945
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777970, upper bound: 5471.9776410
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777229, upper bound: 5471.9776960
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9778062, upper bound: 5471.9776444
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776423, upper bound: 5471.9776988
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776988, upper bound: 5471.9777192
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776444, upper bound: 5471.9778061
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776960, upper bound: 5471.9777229
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9776410, upper bound: 5471.9777970
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777945, upper bound: 5471.9776423
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777229, upper bound: 5471.9777104
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777602, upper bound: 5471.9776444
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 3, lower bound: -5471.9777014, upper bound: 5471.9776994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755311
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755311
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756047
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756046
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755223
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756094, upper bound: 5471.9755210
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756094, upper bound: 5471.9755210
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755496, upper bound: 5471.9755402
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755496, upper bound: 5471.9755402
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755495
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755495
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756094
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756094
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756047, upper bound: 5471.9755210
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756046, upper bound: 5471.9755210
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755311
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755311
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756047
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755223
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756094, upper bound: 5471.9755210
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756094, upper bound: 5471.9755210
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755496, upper bound: 5471.9755402
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755496, upper bound: 5471.9755402
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755495
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755402, upper bound: 5471.9755495
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756094
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755210, upper bound: 5471.9756094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755532, upper bound: 5471.9755568
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755223, upper bound: 5471.9756099
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756099, upper bound: 5471.9755223
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755568, upper bound: 5471.9755532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756047, upper bound: 5471.9755210
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9756046, upper bound: 5471.9755210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754618
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754618
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754684
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755523
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754684
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755523
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755540, upper bound: 5471.9754408
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754408
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755537, upper bound: 5471.9754408
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754883, upper bound: 5471.9754408
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754408
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754883, upper bound: 5471.9754408
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754830, upper bound: 5471.9754408
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754832, upper bound: 5471.9754728
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754830, upper bound: 5471.9754408
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754728
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754832
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754830
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754832
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754830
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754882
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755537
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754883
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755537
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754931
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755540, upper bound: 5471.9754408
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754408
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754920, upper bound: 5471.9754836
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754411
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755523, upper bound: 5471.9754408
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754684, upper bound: 5471.9754408
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755523, upper bound: 5471.9754408
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754684, upper bound: 5471.9754408
time: 1.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755523
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9755540, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9755537, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754883, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754883, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754830, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754832, upper bound: 5471.9754728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754830, upper bound: 5471.9754408
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754832
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754832
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754882
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755537
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754411, upper bound: 5471.9754931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754836, upper bound: 5471.9754920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9755540
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9755540, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754728, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754931, upper bound: 5471.9754411
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754920, upper bound: 5471.9754836
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754411
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754408, upper bound: 5471.9754836
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9755523, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754684, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9755523, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.44
Output dim: 3, lower bound: -5471.9754684, upper bound: 5471.9754408
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 3, lower bound: -5471.9755312, upper bound: 5471.9755402
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=6162.86279296875
rel_dist={3: [-5471.978598948089, 5471.97859894809]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1109.84 seconds
