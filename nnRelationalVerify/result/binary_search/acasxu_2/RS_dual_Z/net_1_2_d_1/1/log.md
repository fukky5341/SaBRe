## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1757.072574941339


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543)
1: (-485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883)
2: (-554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562)
3: (-788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867)
4: (-931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520)

## BASE Result
execution time: IAR + LP analysis = 1.40 + 2.01 = 3.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1270497, upper bound: 1757.1270497


# Binary Search by BASE starts (time budget: 1196.59 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1264512206008, 1757.1264512206008]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2002.608154296875
rel_dist={0: [-1757.1254498169274, 1757.1254498169274]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=2002.608154296875
rel_dist={0: [-1757.1247108341604, 1757.1247108341604]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=2002.608154296875
rel_dist={0: [-1757.1241630696206, 1757.1241630696195]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=2002.608154296875
rel_dist={0: [-1757.1238440599118, 1757.1238440599118]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=2002.608154296875
rel_dist={0: [-1757.1236761183195, 1757.12367608548]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=2002.608154296875
rel_dist={0: [-1757.1235920351874, 1757.1235920868658]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=2002.608154296875
rel_dist={0: [-1757.1235500100406, 1757.123550071139]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=2002.608154296875
rel_dist={0: [-1757.1235290537325, 1757.1235289974684]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=2002.608154296875
rel_dist={0: [-1757.1235185379064, 1757.1235185379064]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=2002.608154296875
rel_dist={0: [-1757.123513279995, 1757.1235132380407]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=2002.608154296875
rel_dist={0: [-1757.1235106114739, 1757.1235106114736]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=2002.608154296875
rel_dist={0: [-1757.123509336572, 1757.1235093365722]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=2002.608154296875
rel_dist={0: [-1757.1235086800148, 1757.1235086800148]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=2002.608154296875
rel_dist={0: [-1757.1235083509137, 1757.1235083132788]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=2002.608154296875
rel_dist={0: [-1757.1235081491786, 1757.123508186512]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=2002.608154296875
rel_dist={0: [-1757.123508107934, 1757.1235080678216]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=2002.608154296875
rel_dist={0: [-1757.1235080660435, 1757.1235080281776]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=2002.608154296875
rel_dist={0: [-1757.1235080494814, 1757.1235080101233]}

## Binary Search Result
Binary search time: 67.25 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1129.34 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1061104
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 1.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.91
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 1.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.85 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1059176
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 4): status=Status.VERIFIED, low=0.9687500, high=1.0000000, mid=0.9687500, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 5) starts
Candidate diff: 0.9843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1059176
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.85
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 5): status=Status.VERIFIED, low=0.9843750, high=1.0000000, mid=0.9843750, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 6) starts
Candidate diff: 0.9921875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1269455
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1061104
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 6): status=Status.VERIFIED, low=0.9921875, high=1.0000000, mid=0.9921875, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 7) starts
Candidate diff: 0.9960938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 7): status=Status.VERIFIED, low=0.9960938, high=1.0000000, mid=0.9960938, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 8) starts
Candidate diff: 0.9980469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 1.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 8): status=Status.VERIFIED, low=0.9980469, high=1.0000000, mid=0.9980469, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 9) starts
Candidate diff: 0.9990234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 9): status=Status.VERIFIED, low=0.9990234, high=1.0000000, mid=0.9990234, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 10) starts
Candidate diff: 0.9995117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 10): status=Status.VERIFIED, low=0.9995117, high=1.0000000, mid=0.9995117, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 11) starts
Candidate diff: 0.9997559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 11): status=Status.VERIFIED, low=0.9997559, high=1.0000000, mid=0.9997559, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 12) starts
Candidate diff: 0.9998779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 12): status=Status.VERIFIED, low=0.9998779, high=1.0000000, mid=0.9998779, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 13) starts
Candidate diff: 0.9999390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 13): status=Status.VERIFIED, low=0.9999390, high=1.0000000, mid=0.9999390, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 14) starts
Candidate diff: 0.9999695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1059176
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 14): status=Status.VERIFIED, low=0.9999695, high=1.0000000, mid=0.9999695, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 15) starts
Candidate diff: 0.9999847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 15): status=Status.VERIFIED, low=0.9999847, high=1.0000000, mid=0.9999847, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 16) starts
Candidate diff: 0.9999924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 16): status=Status.VERIFIED, low=0.9999924, high=1.0000000, mid=0.9999924, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 17) starts
Candidate diff: 0.9999962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1057036, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 17): status=Status.VERIFIED, low=0.9999962, high=1.0000000, mid=0.9999962, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 18) starts
Candidate diff: 0.9999981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.92 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.00
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 18): status=Status.VERIFIED, low=0.9999981, high=1.0000000, mid=0.9999981, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary search (step 19) starts
Candidate diff: 0.9999990


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1270466, upper bound: 1757.1269455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -1757.1269455, upper bound: 1757.1270466

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 0, lower bound: -1757.1061104, upper bound: 1757.1061119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 0, lower bound: -1757.1059176, upper bound: 1757.1063020
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 0, lower bound: -1757.1063020, upper bound: 1757.1059176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 0, lower bound: -1757.1061119, upper bound: 1757.1061104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589
time: 0.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1055999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1056548, upper bound: 1757.1056878
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1058818
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1057036
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1054589, upper bound: 1757.1054589
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1058818, upper bound: 1757.1054589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1056878, upper bound: 1757.1056548
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1055999, upper bound: 1757.1054589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.62
Output dim: 0, lower bound: -1757.0339456, upper bound: 1757.0339456
Binary search (step 19): status=Status.VERIFIED, low=0.9999990, high=1.0000000, mid=0.9999990, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407063]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9999990463256836
execution time: 872.61 seconds
