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
execution time: IAR + LP analysis = 1.88 + 2.37 = 4.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -5471.9793324, upper bound: 5471.9793324


# Binary Search by BASE starts (time budget: 1195.75 seconds, max iter: 100)

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
rel_dist={3: [-5471.977386511122, 5471.977386510684]}

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
rel_dist={3: [-5471.977385892953, 5471.977385893644]}

## Binary Search Result
Binary search time: 86.71 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1109.03 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791374, upper bound: 5471.9791378
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791378, upper bound: 5471.9791374
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.15
Output dim: 3, lower bound: -5471.9791374, upper bound: 5471.9791378
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.15
Output dim: 3, lower bound: -5471.9791378, upper bound: 5471.9791374

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791374, upper bound: 5471.9789520
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789583, upper bound: 5471.9791378
time: 0.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791198, upper bound: 5471.9791361
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791367, upper bound: 5471.9791203
time: 1.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.02
Output dim: 3, lower bound: -5471.9791374, upper bound: 5471.9789520
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.02
Output dim: 3, lower bound: -5471.9789583, upper bound: 5471.9791378
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.02
Output dim: 3, lower bound: -5471.9791198, upper bound: 5471.9791361
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.02
Output dim: 3, lower bound: -5471.9791367, upper bound: 5471.9791203

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790888, upper bound: 5471.9789006
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790985, upper bound: 5471.9788839
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788717, upper bound: 5471.9790020
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788737, upper bound: 5471.9790536
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790598, upper bound: 5471.9790941
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9790829
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788074, upper bound: 5471.9788544
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788073, upper bound: 5471.9788544
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9790888, upper bound: 5471.9789006
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9790985, upper bound: 5471.9788839
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9788717, upper bound: 5471.9790020
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9788737, upper bound: 5471.9790536
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9790598, upper bound: 5471.9790941
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9790829
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9788074, upper bound: 5471.9788544
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 3, lower bound: -5471.9788073, upper bound: 5471.9788544

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9788975
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790648, upper bound: 5471.9789006
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788404
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790036, upper bound: 5471.9788305
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788717, upper bound: 5471.9789198
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788432, upper bound: 5471.9790020
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783189, upper bound: 5471.9784301
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783310, upper bound: 5471.9784302
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780255, upper bound: 5471.9780182
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779846, upper bound: 5471.9780793
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9790625
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790631, upper bound: 5471.9790829
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9773311
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9773311
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788071, upper bound: 5471.9788544
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788073, upper bound: 5471.9788251
time: 1.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9788975
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790648, upper bound: 5471.9789006
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788404
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790036, upper bound: 5471.9788305
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9788717, upper bound: 5471.9789198
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9788432, upper bound: 5471.9790020
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9783189, upper bound: 5471.9784301
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9783310, upper bound: 5471.9784302
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9780255, upper bound: 5471.9780182
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9779846, upper bound: 5471.9780793
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790643, upper bound: 5471.9790625
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9790631, upper bound: 5471.9790829
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9773311
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9773311
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9788071, upper bound: 5471.9788544
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 3, lower bound: -5471.9788073, upper bound: 5471.9788251

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789536, upper bound: 5471.9787832
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789110, upper bound: 5471.9787770
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789456, upper bound: 5471.9788713
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790466, upper bound: 5471.9788755
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790236, upper bound: 5471.9788388
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788404
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784710, upper bound: 5471.9783227
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784710, upper bound: 5471.9783227
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786121, upper bound: 5471.9786464
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786121, upper bound: 5471.9786464
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787215, upper bound: 5471.9788635
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787396, upper bound: 5471.9788449
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781695, upper bound: 5471.9783344
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782283, upper bound: 5471.9783017
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783006, upper bound: 5471.9783201
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783310, upper bound: 5471.9784301
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773206
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773206
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772339, upper bound: 5471.9773186
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772339, upper bound: 5471.9773186
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789813, upper bound: 5471.9790624
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790641, upper bound: 5471.9789772
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789627, upper bound: 5471.9789076
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789158, upper bound: 5471.9789797
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771068, upper bound: 5471.9772046
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771068, upper bound: 5471.9771425
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772038, upper bound: 5471.9773311
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9772085
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788070, upper bound: 5471.9787964
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787334, upper bound: 5471.9788544
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780472, upper bound: 5471.9780583
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779906, upper bound: 5471.9781065
time: 0.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789536, upper bound: 5471.9787832
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789110, upper bound: 5471.9787770
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789456, upper bound: 5471.9788713
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9790466, upper bound: 5471.9788755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9790236, upper bound: 5471.9788388
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788404
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9784710, upper bound: 5471.9783227
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9784710, upper bound: 5471.9783227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9786121, upper bound: 5471.9786464
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9786121, upper bound: 5471.9786464
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9787215, upper bound: 5471.9788635
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9787396, upper bound: 5471.9788449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9781695, upper bound: 5471.9783344
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9782283, upper bound: 5471.9783017
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9783006, upper bound: 5471.9783201
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9783310, upper bound: 5471.9784301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773206
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773206
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772339, upper bound: 5471.9773186
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772339, upper bound: 5471.9773186
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789813, upper bound: 5471.9790624
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9790641, upper bound: 5471.9789772
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789627, upper bound: 5471.9789076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9789158, upper bound: 5471.9789797
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9771068, upper bound: 5471.9772046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9771068, upper bound: 5471.9771425
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772038, upper bound: 5471.9773311
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9772043, upper bound: 5471.9772085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9788070, upper bound: 5471.9787964
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9787334, upper bound: 5471.9788544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9780472, upper bound: 5471.9780583
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 3, lower bound: -5471.9779906, upper bound: 5471.9781065

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789557, upper bound: 5471.9787788
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788666, upper bound: 5471.9787832
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786306, upper bound: 5471.9786267
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787315, upper bound: 5471.9786266
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788951, upper bound: 5471.9788565
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789178, upper bound: 5471.9788566
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790280, upper bound: 5471.9788755
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790355, upper bound: 5471.9788606
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786892, upper bound: 5471.9786892
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787894, upper bound: 5471.9786990
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788311
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790241, upper bound: 5471.9788404
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782817, upper bound: 5471.9781583
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782721, upper bound: 5471.9781583
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783077, upper bound: 5471.9783015
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783335, upper bound: 5471.9783093
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784382, upper bound: 5471.9785137
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784761, upper bound: 5471.9784930
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787215, upper bound: 5471.9788630
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787207, upper bound: 5471.9788635
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772471, upper bound: 5471.9773786
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772471, upper bound: 5471.9773786
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781215, upper bound: 5471.9782810
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9782026
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775563, upper bound: 5471.9775792
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775057, upper bound: 5471.9775762
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780234, upper bound: 5471.9780184
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780346, upper bound: 5471.9781082
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782861
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782861
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772008, upper bound: 5471.9773206
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773014
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771731, upper bound: 5471.9772243
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771476, upper bound: 5471.9772319
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772299, upper bound: 5471.9772264
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772296, upper bound: 5471.9773151
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769227, upper bound: 5471.9770014
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769227, upper bound: 5471.9770014
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789822
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789478, upper bound: 5471.9789822
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789852, upper bound: 5471.9788997
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788372, upper bound: 5471.9788372
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785097, upper bound: 5471.9784230
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785227, upper bound: 5471.9784503
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771002, upper bound: 5471.9771143
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771027, upper bound: 5471.9771343
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770669, upper bound: 5471.9770806
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770648, upper bound: 5471.9771258
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770412, upper bound: 5471.9770412
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770447, upper bound: 5471.9770412
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786136, upper bound: 5471.9786322
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786137, upper bound: 5471.9786322
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787334, upper bound: 5471.9788543
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787206, upper bound: 5471.9788544
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779906, upper bound: 5471.9781052
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779885, upper bound: 5471.9781065
time: 0.86 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9789557, upper bound: 5471.9787788
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9788666, upper bound: 5471.9787832
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786306, upper bound: 5471.9786267
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787315, upper bound: 5471.9786266
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9788951, upper bound: 5471.9788565
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9789178, upper bound: 5471.9788566
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9790280, upper bound: 5471.9788755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9790355, upper bound: 5471.9788606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786892, upper bound: 5471.9786892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787894, upper bound: 5471.9786990
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9790512, upper bound: 5471.9788311
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9790241, upper bound: 5471.9788404
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9782817, upper bound: 5471.9781583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9782721, upper bound: 5471.9781583
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9783077, upper bound: 5471.9783015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9783335, upper bound: 5471.9783093
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9784382, upper bound: 5471.9785137
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9784761, upper bound: 5471.9784930
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787215, upper bound: 5471.9788630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787207, upper bound: 5471.9788635
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772471, upper bound: 5471.9773786
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772471, upper bound: 5471.9773786
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9781215, upper bound: 5471.9782810
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9782026
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9775563, upper bound: 5471.9775792
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9775057, upper bound: 5471.9775762
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9780234, upper bound: 5471.9780184
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9780346, upper bound: 5471.9781082
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772008, upper bound: 5471.9773206
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772398, upper bound: 5471.9773014
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9771731, upper bound: 5471.9772243
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9771476, upper bound: 5471.9772319
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772299, upper bound: 5471.9772264
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9772296, upper bound: 5471.9773151
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9769227, upper bound: 5471.9770014
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9769227, upper bound: 5471.9770014
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789822
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9789478, upper bound: 5471.9789822
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9789852, upper bound: 5471.9788997
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9788372, upper bound: 5471.9788372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9785097, upper bound: 5471.9784230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9785227, upper bound: 5471.9784503
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9771002, upper bound: 5471.9771143
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9771027, upper bound: 5471.9771343
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9770669, upper bound: 5471.9770806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9770648, upper bound: 5471.9771258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9770412, upper bound: 5471.9770412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9770447, upper bound: 5471.9770412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786136, upper bound: 5471.9786322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9786137, upper bound: 5471.9786322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787334, upper bound: 5471.9788543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9787206, upper bound: 5471.9788544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9779906, upper bound: 5471.9781052
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 3, lower bound: -5471.9779885, upper bound: 5471.9781065

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788905, upper bound: 5471.9785472
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785679, upper bound: 5471.9785414
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784032, upper bound: 5471.9783662
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783283, upper bound: 5471.9783398
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785785, upper bound: 5471.9785809
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785785, upper bound: 5471.9785804
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782123, upper bound: 5471.9781925
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782123, upper bound: 5471.9781925
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787743, upper bound: 5471.9787833
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788418, upper bound: 5471.9787743
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780010, upper bound: 5471.9779760
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779472, upper bound: 5471.9779760
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789633, upper bound: 5471.9788014
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789656, upper bound: 5471.9787828
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780579, upper bound: 5471.9779399
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779399, upper bound: 5471.9779399
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786106, upper bound: 5471.9785490
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785490, upper bound: 5471.9785490
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779243, upper bound: 5471.9778698
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779228, upper bound: 5471.9779212
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787944, upper bound: 5471.9786895
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788104, upper bound: 5471.9786895
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790221, upper bound: 5471.9788404
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790240, upper bound: 5471.9788322
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724311, upper bound: 5471.9724311
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9724311, upper bound: 5471.9724311
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782817, upper bound: 5471.9781583
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781583, upper bound: 5471.9781583
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782597, upper bound: 5471.9781340
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781417, upper bound: 5471.9781340
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782021, upper bound: 5471.9781963
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782044, upper bound: 5471.9781963
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781711, upper bound: 5471.9781552
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781776, upper bound: 5471.9781504
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780525, upper bound: 5471.9781542
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780525, upper bound: 5471.9781542
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784743, upper bound: 5471.9784847
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784761, upper bound: 5471.9784931
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9787437
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787099, upper bound: 5471.9788630
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786631, upper bound: 5471.9788039
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786621, upper bound: 5471.9788127
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754517, upper bound: 5471.9753416
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754517, upper bound: 5471.9753416
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772249, upper bound: 5471.9773731
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772468, upper bound: 5471.9773252
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781116, upper bound: 5471.9782800
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781087, upper bound: 5471.9782672
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9782026
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9781691
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767632, upper bound: 5471.9767259
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767632, upper bound: 5471.9767480
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768475, upper bound: 5471.9768475
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768942, upper bound: 5471.9768585
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777495, upper bound: 5471.9777495
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777495, upper bound: 5471.9777495
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780320, upper bound: 5471.9781082
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780346, upper bound: 5471.9781074
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782461
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782861
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782461
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782861
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771858, upper bound: 5471.9773106
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771858, upper bound: 5471.9771858
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759995, upper bound: 5471.9760309
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759995, upper bound: 5471.9760309
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771610, upper bound: 5471.9771915
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771651, upper bound: 5471.9771437
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770690, upper bound: 5471.9770967
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770690, upper bound: 5471.9770967
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760577, upper bound: 5471.9760577
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760577, upper bound: 5471.9760577
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771856, upper bound: 5471.9773004
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771839, upper bound: 5471.9773004
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768125, upper bound: 5471.9768763
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768125, upper bound: 5471.9768363
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767589, upper bound: 5471.9768716
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767633, upper bound: 5471.9767691
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789577
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789822
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788028, upper bound: 5471.9788745
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788060, upper bound: 5471.9788669
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788359, upper bound: 5471.9788359
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789852, upper bound: 5471.9788997
time: 0.99 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788905, upper bound: 5471.9785472
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9785679, upper bound: 5471.9785414
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9784032, upper bound: 5471.9783662
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9783283, upper bound: 5471.9783398
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9785785, upper bound: 5471.9785809
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9785785, upper bound: 5471.9785804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782123, upper bound: 5471.9781925
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782123, upper bound: 5471.9781925
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9787743, upper bound: 5471.9787833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788418, upper bound: 5471.9787743
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780010, upper bound: 5471.9779760
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9779472, upper bound: 5471.9779760
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9789633, upper bound: 5471.9788014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9789656, upper bound: 5471.9787828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780579, upper bound: 5471.9779399
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9779399, upper bound: 5471.9779399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9786106, upper bound: 5471.9785490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9785490, upper bound: 5471.9785490
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9779243, upper bound: 5471.9778698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9779228, upper bound: 5471.9779212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9787944, upper bound: 5471.9786895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788104, upper bound: 5471.9786895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9790221, upper bound: 5471.9788404
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9790240, upper bound: 5471.9788322
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9724447, upper bound: 5471.9724447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9724311, upper bound: 5471.9724311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9724311, upper bound: 5471.9724311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782817, upper bound: 5471.9781583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781583, upper bound: 5471.9781583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782597, upper bound: 5471.9781340
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781417, upper bound: 5471.9781340
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782021, upper bound: 5471.9781963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782044, upper bound: 5471.9781963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781711, upper bound: 5471.9781552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781776, upper bound: 5471.9781504
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780525, upper bound: 5471.9781542
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780525, upper bound: 5471.9781542
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9784743, upper bound: 5471.9784847
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9784761, upper bound: 5471.9784931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9787437
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9787099, upper bound: 5471.9788630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9786631, upper bound: 5471.9788039
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9786621, upper bound: 5471.9788127
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9754517, upper bound: 5471.9753416
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9754517, upper bound: 5471.9753416
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9772249, upper bound: 5471.9773731
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9772468, upper bound: 5471.9773252
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781116, upper bound: 5471.9782800
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781087, upper bound: 5471.9782672
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9782026
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9781189, upper bound: 5471.9781691
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9767632, upper bound: 5471.9767259
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9767632, upper bound: 5471.9767480
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9768475, upper bound: 5471.9768475
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9768942, upper bound: 5471.9768585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9777495, upper bound: 5471.9777495
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9777495, upper bound: 5471.9777495
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780320, upper bound: 5471.9781082
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9780346, upper bound: 5471.9781074
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782860, upper bound: 5471.9782461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782461
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9782415, upper bound: 5471.9782861
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771858, upper bound: 5471.9773106
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771858, upper bound: 5471.9771858
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9759995, upper bound: 5471.9760309
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9759995, upper bound: 5471.9760309
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771610, upper bound: 5471.9771915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771651, upper bound: 5471.9771437
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9770690, upper bound: 5471.9770967
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9770690, upper bound: 5471.9770967
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9760577, upper bound: 5471.9760577
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9760577, upper bound: 5471.9760577
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771856, upper bound: 5471.9773004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9771839, upper bound: 5471.9773004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9768125, upper bound: 5471.9768763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9768125, upper bound: 5471.9768363
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9767589, upper bound: 5471.9768716
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9767633, upper bound: 5471.9767691
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789577
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9789165, upper bound: 5471.9789822
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788028, upper bound: 5471.9788745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788060, upper bound: 5471.9788669
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9788359, upper bound: 5471.9788359
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.00
Output dim: 3, lower bound: -5471.9789852, upper bound: 5471.9788997
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9788372, upper bound: 5471.9788372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9785097, upper bound: 5471.9784230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9785227, upper bound: 5471.9784503
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9786085, upper bound: 5471.9786271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9728016, upper bound: 5471.9728017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9771002, upper bound: 5471.9771143
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9771027, upper bound: 5471.9771343
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9770669, upper bound: 5471.9770806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9770648, upper bound: 5471.9771258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9770412, upper bound: 5471.9770412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9770447, upper bound: 5471.9770412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9786136, upper bound: 5471.9786322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9786137, upper bound: 5471.9786322
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9787334, upper bound: 5471.9788543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9787206, upper bound: 5471.9788544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9750527, upper bound: 5471.9751173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9779906, upper bound: 5471.9781052
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.00
Output dim: 3, lower bound: -5471.9779885, upper bound: 5471.9781065
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=6162.86279296875
rel_dist={3: [-5471.979332446701, 5471.979332446701]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788568, upper bound: 5471.9788519
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788519, upper bound: 5471.9788568
time: 1.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 3, lower bound: -5471.9788568, upper bound: 5471.9788519
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 3, lower bound: -5471.9788519, upper bound: 5471.9788568

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
time: 1.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787632, upper bound: 5471.9787624
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787387, upper bound: 5471.9787989
time: 1.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -5471.9787632, upper bound: 5471.9787624
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 3, lower bound: -5471.9787387, upper bound: 5471.9787989

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786240, upper bound: 5471.9785871
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786240, upper bound: 5471.9785869
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787605, upper bound: 5471.9787583
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786807, upper bound: 5471.9786702
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786876, upper bound: 5471.9786509
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787386, upper bound: 5471.9787897
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787262, upper bound: 5471.9787897
time: 1.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9786240, upper bound: 5471.9785871
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9786240, upper bound: 5471.9785869
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9787605, upper bound: 5471.9787583
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9787606, upper bound: 5471.9787589
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9786807, upper bound: 5471.9786702
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9786876, upper bound: 5471.9786509
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9787386, upper bound: 5471.9787897
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 3, lower bound: -5471.9787262, upper bound: 5471.9787897

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785751
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786154, upper bound: 5471.9785341
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785692, upper bound: 5471.9785779
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785692, upper bound: 5471.9785857
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787127, upper bound: 5471.9787132
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787167, upper bound: 5471.9787114
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787599, upper bound: 5471.9787589
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787605, upper bound: 5471.9787504
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782980
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782980
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786406, upper bound: 5471.9786211
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786503, upper bound: 5471.9786083
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786517, upper bound: 5471.9785860
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785807, upper bound: 5471.9786924
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787012, upper bound: 5471.9785828
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785717, upper bound: 5471.9787892
time: 1.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785751
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9786154, upper bound: 5471.9785341
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9785692, upper bound: 5471.9785779
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9785692, upper bound: 5471.9785857
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9787127, upper bound: 5471.9787132
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9787167, upper bound: 5471.9787114
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9787599, upper bound: 5471.9787589
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9787605, upper bound: 5471.9787504
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782980
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782980
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9786406, upper bound: 5471.9786211
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9786503, upper bound: 5471.9786083
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9786517, upper bound: 5471.9785860
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9785807, upper bound: 5471.9786924
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9787012, upper bound: 5471.9785828
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 3, lower bound: -5471.9785717, upper bound: 5471.9787892

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785595
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786093, upper bound: 5471.9785752
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785049, upper bound: 5471.9784825
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785495, upper bound: 5471.9784712
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785016, upper bound: 5471.9785602
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786156, upper bound: 5471.9785016
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785691, upper bound: 5471.9785446
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785563, upper bound: 5471.9785857
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782737
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782737
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786972, upper bound: 5471.9787049
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787157, upper bound: 5471.9786919
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782510, upper bound: 5471.9783086
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783161, upper bound: 5471.9782695
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779990, upper bound: 5471.9779559
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780910, upper bound: 5471.9779560
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782610, upper bound: 5471.9782302
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782603, upper bound: 5471.9781867
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782980
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782613
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783660, upper bound: 5471.9784516
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784074, upper bound: 5471.9784516
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786178, upper bound: 5471.9785864
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785888, upper bound: 5471.9785610
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786517, upper bound: 5471.9785480
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786323, upper bound: 5471.9785860
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784783, upper bound: 5471.9786270
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784394, upper bound: 5471.9785990
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780092, upper bound: 5471.9779137
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779759, upper bound: 5471.9779137
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784654, upper bound: 5471.9787029
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784937, upper bound: 5471.9787024
time: 0.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785595
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786093, upper bound: 5471.9785752
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785049, upper bound: 5471.9784825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785495, upper bound: 5471.9784712
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785016, upper bound: 5471.9785602
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786156, upper bound: 5471.9785016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785691, upper bound: 5471.9785446
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785563, upper bound: 5471.9785857
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782737
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786972, upper bound: 5471.9787049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9787157, upper bound: 5471.9786919
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9782510, upper bound: 5471.9783086
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9783161, upper bound: 5471.9782695
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9779990, upper bound: 5471.9779559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9780910, upper bound: 5471.9779560
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9782610, upper bound: 5471.9782302
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9782603, upper bound: 5471.9781867
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782980
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782613
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9783660, upper bound: 5471.9784516
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9784074, upper bound: 5471.9784516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786178, upper bound: 5471.9785864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9785888, upper bound: 5471.9785610
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786517, upper bound: 5471.9785480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9786323, upper bound: 5471.9785860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9784783, upper bound: 5471.9786270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9784394, upper bound: 5471.9785990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9780092, upper bound: 5471.9779137
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9779759, upper bound: 5471.9779137
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9784654, upper bound: 5471.9787029
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 3, lower bound: -5471.9784937, upper bound: 5471.9787024

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785427
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786086, upper bound: 5471.9785595
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782985, upper bound: 5471.9782532
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782985, upper bound: 5471.9782532
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785049, upper bound: 5471.9784805
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784589, upper bound: 5471.9784825
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785488, upper bound: 5471.9784447
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784780, upper bound: 5471.9784590
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782765, upper bound: 5471.9782597
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782766, upper bound: 5471.9782597
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771543, upper bound: 5471.9770123
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771543, upper bound: 5471.9770123
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785142, upper bound: 5471.9784749
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784868, upper bound: 5471.9784787
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785385, upper bound: 5471.9785857
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785385, upper bound: 5471.9785838
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782667
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782747, upper bound: 5471.9782737
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782587, upper bound: 5471.9782737
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783058, upper bound: 5471.9782680
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785927, upper bound: 5471.9785159
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785927, upper bound: 5471.9785159
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785480, upper bound: 5471.9786209
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786516, upper bound: 5471.9786209
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781958, upper bound: 5471.9782676
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782078, upper bound: 5471.9782335
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782905, upper bound: 5471.9782529
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782905, upper bound: 5471.9782528
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780884, upper bound: 5471.9779271
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780259, upper bound: 5471.9779559
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780114, upper bound: 5471.9778675
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779961, upper bound: 5471.9778603
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774164, upper bound: 5471.9773191
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774163, upper bound: 5471.9773191
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780632, upper bound: 5471.9779136
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779643, upper bound: 5471.9779136
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782730
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782661, upper bound: 5471.9782980
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782602
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783206, upper bound: 5471.9782602
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783065, upper bound: 5471.9783898
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782963, upper bound: 5471.9783872
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778445, upper bound: 5471.9779142
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778319, upper bound: 5471.9779142
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782963, upper bound: 5471.9783748
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783196, upper bound: 5471.9783748
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781283, upper bound: 5471.9781349
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781283, upper bound: 5471.9781349
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785062, upper bound: 5471.9785218
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785468, upper bound: 5471.9784904
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785974, upper bound: 5471.9785084
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785094, upper bound: 5471.9785512
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783867, upper bound: 5471.9785589
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783630, upper bound: 5471.9785172
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775219, upper bound: 5471.9775380
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775219, upper bound: 5471.9775380
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778718, upper bound: 5471.9778593
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779825, upper bound: 5471.9778706
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779547, upper bound: 5471.9778342
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779162, upper bound: 5471.9778845
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781781, upper bound: 5471.9784463
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781781, upper bound: 5471.9784463
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784137, upper bound: 5471.9786409
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784422, upper bound: 5471.9785868
time: 1.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9786110, upper bound: 5471.9785427
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9786086, upper bound: 5471.9785595
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782985, upper bound: 5471.9782532
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782985, upper bound: 5471.9782532
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785049, upper bound: 5471.9784805
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9784589, upper bound: 5471.9784825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785488, upper bound: 5471.9784447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9784780, upper bound: 5471.9784590
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782765, upper bound: 5471.9782597
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782766, upper bound: 5471.9782597
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9771543, upper bound: 5471.9770123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9771543, upper bound: 5471.9770123
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785142, upper bound: 5471.9784749
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9784868, upper bound: 5471.9784787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785385, upper bound: 5471.9785857
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785385, upper bound: 5471.9785838
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783115, upper bound: 5471.9782667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782747, upper bound: 5471.9782737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782587, upper bound: 5471.9782737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783058, upper bound: 5471.9782680
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785927, upper bound: 5471.9785159
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785927, upper bound: 5471.9785159
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785480, upper bound: 5471.9786209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9786516, upper bound: 5471.9786209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9781958, upper bound: 5471.9782676
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782078, upper bound: 5471.9782335
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782905, upper bound: 5471.9782529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782905, upper bound: 5471.9782528
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9780884, upper bound: 5471.9779271
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9780259, upper bound: 5471.9779559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9780114, upper bound: 5471.9778675
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9779961, upper bound: 5471.9778603
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9774164, upper bound: 5471.9773191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9774163, upper bound: 5471.9773191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9780632, upper bound: 5471.9779136
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9779643, upper bound: 5471.9779136
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782730
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782661, upper bound: 5471.9782980
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783302, upper bound: 5471.9782602
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783206, upper bound: 5471.9782602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783065, upper bound: 5471.9783898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782963, upper bound: 5471.9783872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9778445, upper bound: 5471.9779142
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9778319, upper bound: 5471.9779142
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9782963, upper bound: 5471.9783748
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783196, upper bound: 5471.9783748
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9781283, upper bound: 5471.9781349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9781283, upper bound: 5471.9781349
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785062, upper bound: 5471.9785218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785468, upper bound: 5471.9784904
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785974, upper bound: 5471.9785084
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9785094, upper bound: 5471.9785512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783867, upper bound: 5471.9785589
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9783630, upper bound: 5471.9785172
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9775219, upper bound: 5471.9775380
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9775219, upper bound: 5471.9775380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9778718, upper bound: 5471.9778593
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9779825, upper bound: 5471.9778706
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9779547, upper bound: 5471.9778342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9779162, upper bound: 5471.9778845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9781781, upper bound: 5471.9784463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9781781, upper bound: 5471.9784463
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9784137, upper bound: 5471.9786409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 3, lower bound: -5471.9784422, upper bound: 5471.9785868

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785052, upper bound: 5471.9785425
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785052, upper bound: 5471.9785427
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775260, upper bound: 5471.9775691
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775260, upper bound: 5471.9775657
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782985, upper bound: 5471.9782124
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782879, upper bound: 5471.9782532
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782567, upper bound: 5471.9782532
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782941, upper bound: 5471.9782080
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784646, upper bound: 5471.9784588
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784843, upper bound: 5471.9784805
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783869, upper bound: 5471.9784156
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783869, upper bound: 5471.9783868
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784447, upper bound: 5471.9784447
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785440, upper bound: 5471.9784447
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784780, upper bound: 5471.9784590
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784447, upper bound: 5471.9784447
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781647, upper bound: 5471.9782259
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781647, upper bound: 5471.9782259
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779444, upper bound: 5471.9779628
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779444, upper bound: 5471.9779625
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9734232, upper bound: 5471.9734232
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9734232, upper bound: 5471.9734232
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771418, upper bound: 5471.9770123
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771370, upper bound: 5471.9770123
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784502, upper bound: 5471.9784502
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784909, upper bound: 5471.9784502
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783964, upper bound: 5471.9783920
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784056, upper bound: 5471.9783890
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770505, upper bound: 5471.9770697
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9771182, upper bound: 5471.9770706
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780518, upper bound: 5471.9780504
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780504, upper bound: 5471.9780504
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780727, upper bound: 5471.9780160
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780727, upper bound: 5471.9780160
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782743, upper bound: 5471.9782737
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782747, upper bound: 5471.9782587
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781838, upper bound: 5471.9782051
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781837, upper bound: 5471.9781838
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780814, upper bound: 5471.9780466
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780274, upper bound: 5471.9780273
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785477, upper bound: 5471.9784756
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785078, upper bound: 5471.9784756
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785870, upper bound: 5471.9785100
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785100, upper bound: 5471.9785100
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785477, upper bound: 5471.9785666
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785477, upper bound: 5471.9785477
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785191, upper bound: 5471.9785363
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786307, upper bound: 5471.9786061
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780575, upper bound: 5471.9780603
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780499, upper bound: 5471.9781616
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775425, upper bound: 5471.9774966
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775425, upper bound: 5471.9774966
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781706, upper bound: 5471.9780477
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780898, upper bound: 5471.9781067
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781637, upper bound: 5471.9781146
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781683, upper bound: 5471.9780760
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774394, upper bound: 5471.9774394
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774431, upper bound: 5471.9774394
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778816, upper bound: 5471.9778907
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779866, upper bound: 5471.9779061
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780114, upper bound: 5471.9778611
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779564, upper bound: 5471.9778675
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756427, upper bound: 5471.9755904
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756427, upper bound: 5471.9755904
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772480, upper bound: 5471.9773182
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774083, upper bound: 5471.9772953
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773068, upper bound: 5471.9772098
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773035, upper bound: 5471.9771686
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770847, upper bound: 5471.9769467
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769467, upper bound: 5471.9769467
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778325, upper bound: 5471.9777741
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778454, upper bound: 5471.9777741
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782043, upper bound: 5471.9781965
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781852, upper bound: 5471.9781852
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781296, upper bound: 5471.9781972
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781296, upper bound: 5471.9781436
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782801, upper bound: 5471.9782053
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782180, upper bound: 5471.9782053
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 2.05 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=6162.86279296875
rel_dist={3: [-5471.979020840545, 5471.979020840547]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785984, upper bound: 5471.9784294
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784295, upper bound: 5471.9785984
time: 1.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 3, lower bound: -5471.9785984, upper bound: 5471.9784294
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 3, lower bound: -5471.9784295, upper bound: 5471.9785984

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785122, upper bound: 5471.9784003
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785676, upper bound: 5471.9783846
time: 0.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782622, upper bound: 5471.9785073
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783325, upper bound: 5471.9784308
time: 1.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -5471.9785122, upper bound: 5471.9784003
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -5471.9785676, upper bound: 5471.9783846
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -5471.9782622, upper bound: 5471.9785073
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 3, lower bound: -5471.9783325, upper bound: 5471.9784308

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784417, upper bound: 5471.9783972
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785052, upper bound: 5471.9783493
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785064, upper bound: 5471.9783222
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784200, upper bound: 5471.9783396
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782319, upper bound: 5471.9785073
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782622, upper bound: 5471.9784818
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782969, upper bound: 5471.9783173
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782933, upper bound: 5471.9783173
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9784417, upper bound: 5471.9783972
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9785052, upper bound: 5471.9783493
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9785064, upper bound: 5471.9783222
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9784200, upper bound: 5471.9783396
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9782319, upper bound: 5471.9785073
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9782622, upper bound: 5471.9784818
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9782969, upper bound: 5471.9783173
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.01
Output dim: 3, lower bound: -5471.9782933, upper bound: 5471.9783173

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783479, upper bound: 5471.9783278
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784417, upper bound: 5471.9783972
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784991, upper bound: 5471.9783355
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784923, upper bound: 5471.9783355
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785065, upper bound: 5471.9783048
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785047, upper bound: 5471.9783223
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783570, upper bound: 5471.9783396
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783570, upper bound: 5471.9783343
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781595, upper bound: 5471.9784425
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781707, upper bound: 5471.9784208
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782186, upper bound: 5471.9784283
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782077, upper bound: 5471.9784248
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778034, upper bound: 5471.9778385
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778145, upper bound: 5471.9778384
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9783075
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782787, upper bound: 5471.9783173
time: 1.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9783479, upper bound: 5471.9783278
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9784417, upper bound: 5471.9783972
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9784991, upper bound: 5471.9783355
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9784923, upper bound: 5471.9783355
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9785065, upper bound: 5471.9783048
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9785047, upper bound: 5471.9783223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9783570, upper bound: 5471.9783396
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9783570, upper bound: 5471.9783343
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9781595, upper bound: 5471.9784425
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9781707, upper bound: 5471.9784208
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9782186, upper bound: 5471.9784283
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9782077, upper bound: 5471.9784248
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9778034, upper bound: 5471.9778385
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9778145, upper bound: 5471.9778384
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9783075
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 3, lower bound: -5471.9782787, upper bound: 5471.9783173

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783283, upper bound: 5471.9783076
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783251, upper bound: 5471.9783247
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777676, upper bound: 5471.9777670
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777575, upper bound: 5471.9777670
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784777, upper bound: 5471.9782893
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783297, upper bound: 5471.9782580
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784617, upper bound: 5471.9783180
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784644, upper bound: 5471.9782832
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784893, upper bound: 5471.9783048
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785064, upper bound: 5471.9783047
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783333, upper bound: 5471.9782306
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784236, upper bound: 5471.9781651
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783166, upper bound: 5471.9783307
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782989, upper bound: 5471.9783396
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782601
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782990, upper bound: 5471.9782625
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780836, upper bound: 5471.9783054
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780836, upper bound: 5471.9783066
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781580, upper bound: 5471.9783497
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781446, upper bound: 5471.9783995
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9779569
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9780334
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769688
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769690
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773760, upper bound: 5471.9775316
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774689, upper bound: 5471.9775488
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777144, upper bound: 5471.9778384
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777223, upper bound: 5471.9778352
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782556, upper bound: 5471.9783066
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9782044
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782518, upper bound: 5471.9782864
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782567, upper bound: 5471.9782688
time: 1.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9783283, upper bound: 5471.9783076
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9783251, upper bound: 5471.9783247
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9777676, upper bound: 5471.9777670
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9777575, upper bound: 5471.9777670
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9784777, upper bound: 5471.9782893
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9783297, upper bound: 5471.9782580
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9784617, upper bound: 5471.9783180
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9784644, upper bound: 5471.9782832
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9784893, upper bound: 5471.9783048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9785064, upper bound: 5471.9783047
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9783333, upper bound: 5471.9782306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9784236, upper bound: 5471.9781651
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9783166, upper bound: 5471.9783307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782989, upper bound: 5471.9783396
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782832, upper bound: 5471.9782601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782990, upper bound: 5471.9782625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9780836, upper bound: 5471.9783054
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9780836, upper bound: 5471.9783066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9781580, upper bound: 5471.9783497
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9781446, upper bound: 5471.9783995
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9779569
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9780334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769690
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9773760, upper bound: 5471.9775316
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9774689, upper bound: 5471.9775488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9777144, upper bound: 5471.9778384
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9777223, upper bound: 5471.9778352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782556, upper bound: 5471.9783066
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9782044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782518, upper bound: 5471.9782864
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.87
Output dim: 3, lower bound: -5471.9782567, upper bound: 5471.9782688

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782841, upper bound: 5471.9782841
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782841, upper bound: 5471.9782841
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782464, upper bound: 5471.9782269
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782464, upper bound: 5471.9782076
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777507, upper bound: 5471.9777227
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777240, upper bound: 5471.9777254
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776323, upper bound: 5471.9776471
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776321, upper bound: 5471.9776330
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777088, upper bound: 5471.9776650
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777031, upper bound: 5471.9776650
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783060, upper bound: 5471.9782493
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783068, upper bound: 5471.9782493
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781623, upper bound: 5471.9780395
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781623, upper bound: 5471.9780395
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784470, upper bound: 5471.9782589
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784474, upper bound: 5471.9782665
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776174, upper bound: 5471.9773937
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774454, upper bound: 5471.9773937
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785054, upper bound: 5471.9783047
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784073, upper bound: 5471.9783047
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783333, upper bound: 5471.9782307
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781630, upper bound: 5471.9781744
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782720, upper bound: 5471.9781652
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782729, upper bound: 5471.9781626
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783546, upper bound: 5471.9783307
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783539, upper bound: 5471.9783209
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773360, upper bound: 5471.9773338
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773338, upper bound: 5471.9773338
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781577, upper bound: 5471.9781602
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782284, upper bound: 5471.9781877
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783240, upper bound: 5471.9782173
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783253, upper bound: 5471.9782596
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780787, upper bound: 5471.9782974
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780787, upper bound: 5471.9782981
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777017, upper bound: 5471.9779527
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776970, upper bound: 5471.9779527
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778806, upper bound: 5471.9780536
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778806, upper bound: 5471.9780536
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777456, upper bound: 5471.9779468
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777450, upper bound: 5471.9779461
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9779569
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778261, upper bound: 5471.9779451
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778256, upper bound: 5471.9779139
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778411, upper bound: 5471.9780266
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769738, upper bound: 5471.9769688
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769688
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758718, upper bound: 5471.9758723
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758718, upper bound: 5471.9758723
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773718, upper bound: 5471.9775316
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774761, upper bound: 5471.9775061
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773339, upper bound: 5471.9775322
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774689, upper bound: 5471.9775334
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775039, upper bound: 5471.9774484
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775045, upper bound: 5471.9774582
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
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778122, upper bound: 5471.9778283
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778145, upper bound: 5471.9778352
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780519, upper bound: 5471.9780726
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780483, upper bound: 5471.9780726
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781540, upper bound: 5471.9781669
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9782038
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780120, upper bound: 5471.9780375
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779569, upper bound: 5471.9780376
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767988, upper bound: 5471.9768166
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767766, upper bound: 5471.9768166
time: 1.18 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782841, upper bound: 5471.9782841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782841, upper bound: 5471.9782841
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782464, upper bound: 5471.9782269
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782464, upper bound: 5471.9782076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777507, upper bound: 5471.9777227
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777240, upper bound: 5471.9777254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9776323, upper bound: 5471.9776471
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9776321, upper bound: 5471.9776330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777088, upper bound: 5471.9776650
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777031, upper bound: 5471.9776650
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783060, upper bound: 5471.9782493
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783068, upper bound: 5471.9782493
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9781623, upper bound: 5471.9780395
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9781623, upper bound: 5471.9780395
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9784470, upper bound: 5471.9782589
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9784474, upper bound: 5471.9782665
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9776174, upper bound: 5471.9773937
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9774454, upper bound: 5471.9773937
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9785054, upper bound: 5471.9783047
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9784073, upper bound: 5471.9783047
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783333, upper bound: 5471.9782307
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9781630, upper bound: 5471.9781744
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782720, upper bound: 5471.9781652
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782729, upper bound: 5471.9781626
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783546, upper bound: 5471.9783307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783539, upper bound: 5471.9783209
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9773360, upper bound: 5471.9773338
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9773338, upper bound: 5471.9773338
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9781577, upper bound: 5471.9781602
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782284, upper bound: 5471.9781877
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783240, upper bound: 5471.9782173
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9783253, upper bound: 5471.9782596
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9780787, upper bound: 5471.9782974
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9780787, upper bound: 5471.9782981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777017, upper bound: 5471.9779527
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9776970, upper bound: 5471.9779527
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778806, upper bound: 5471.9780536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778806, upper bound: 5471.9780536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777456, upper bound: 5471.9779468
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9777450, upper bound: 5471.9779461
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778414, upper bound: 5471.9779569
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778261, upper bound: 5471.9779451
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778256, upper bound: 5471.9779139
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778411, upper bound: 5471.9780266
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9769738, upper bound: 5471.9769688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9769913, upper bound: 5471.9769688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9758718, upper bound: 5471.9758723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9758718, upper bound: 5471.9758723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9773718, upper bound: 5471.9775316
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9774761, upper bound: 5471.9775061
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9773339, upper bound: 5471.9775322
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9774689, upper bound: 5471.9775334
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9775039, upper bound: 5471.9774484
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9775045, upper bound: 5471.9774582
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778122, upper bound: 5471.9778283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9778145, upper bound: 5471.9778352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9780519, upper bound: 5471.9780726
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9780483, upper bound: 5471.9780726
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9781540, upper bound: 5471.9781669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9782780, upper bound: 5471.9782038
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9780120, upper bound: 5471.9780375
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9779569, upper bound: 5471.9780376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9767988, upper bound: 5471.9768166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 3, lower bound: -5471.9767766, upper bound: 5471.9768166

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776716, upper bound: 5471.9776716
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776716, upper bound: 5471.9776716
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782480, upper bound: 5471.9782480
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782480, upper bound: 5471.9782480
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344
1: -1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438
2: -1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969
3: -1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930
4: -1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016

Time for backsubstitution: 1.92 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=6162.86279296875
rel_dist={3: [-5471.978598948089, 5471.97859894809]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1110.43 seconds
