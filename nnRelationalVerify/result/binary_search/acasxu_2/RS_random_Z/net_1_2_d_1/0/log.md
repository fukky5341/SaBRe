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
execution time: IAR + LP analysis = 1.21 + 2.39 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1270497, upper bound: 1757.1270497


# Binary Search by BASE starts (time budget: 1196.40 seconds, max iter: 100)

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
Binary search time: 63.72 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1132.68 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1268226, upper bound: 1757.1266047
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1266047, upper bound: 1757.1268226
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -1757.1268226, upper bound: 1757.1266047
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 0, lower bound: -1757.1266047, upper bound: 1757.1268226

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249622, upper bound: 1757.1249585
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249585, upper bound: 1757.1249585
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1266047, upper bound: 1757.1265500
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1268226
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1757.1249622, upper bound: 1757.1249585
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1757.1249585, upper bound: 1757.1249585
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1757.1266047, upper bound: 1757.1265500
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1268226

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239750, upper bound: 1757.1239370
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239500
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249622, upper bound: 1757.1249560
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249597, upper bound: 1757.1249583
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261322, upper bound: 1757.1261555
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1262813, upper bound: 1757.1259840
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263918, upper bound: 1757.1266631
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1268225
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1239750, upper bound: 1757.1239370
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239500
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1249622, upper bound: 1757.1249560
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1249597, upper bound: 1757.1249583
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1261322, upper bound: 1757.1261555
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1262813, upper bound: 1757.1259840
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1263918, upper bound: 1757.1266631
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1268225

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239370
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239370
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260868, upper bound: 1757.1261213
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259410, upper bound: 1757.1260681
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1048472, upper bound: 1757.1044277
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044673, upper bound: 1757.1044277
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1205978, upper bound: 1757.1205978
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1205978, upper bound: 1757.1205978
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1264276
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267816
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239370
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1239370, upper bound: 1757.1239370
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1260868, upper bound: 1757.1261213
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1259410, upper bound: 1757.1260681
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1048472, upper bound: 1757.1044277
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1044673, upper bound: 1757.1044277
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1205978, upper bound: 1757.1205978
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1205978, upper bound: 1757.1205978
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1264045, upper bound: 1757.1264276
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239246, upper bound: 1757.1239234
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1202293, upper bound: 1757.1202293
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1202293, upper bound: 1757.1202293
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235668, upper bound: 1757.1235753
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235668, upper bound: 1757.1235668
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259410, upper bound: 1757.1259410
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260868, upper bound: 1757.1261213
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0450510, upper bound: 1757.0450510
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0450510, upper bound: 1757.0450510
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043396, upper bound: 1757.1043396
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044451, upper bound: 1757.1043396
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028354, upper bound: 1757.1028354
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028523, upper bound: 1757.1030495
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259103, upper bound: 1757.1257686
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258839, upper bound: 1757.1258550
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267816
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267584
time: 0.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1239246, upper bound: 1757.1239234
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1239234, upper bound: 1757.1239234
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1202293, upper bound: 1757.1202293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1202293, upper bound: 1757.1202293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1235668, upper bound: 1757.1235753
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1235668, upper bound: 1757.1235668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1259410, upper bound: 1757.1259410
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1260868, upper bound: 1757.1261213
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0450510, upper bound: 1757.0450510
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.0450510, upper bound: 1757.0450510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1043396, upper bound: 1757.1043396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1044451, upper bound: 1757.1043396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1028354, upper bound: 1757.1028354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1028523, upper bound: 1757.1030495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1259103, upper bound: 1757.1257686
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1258839, upper bound: 1757.1258550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267816
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -1757.1263485, upper bound: 1757.1267584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237515, upper bound: 1757.1237473
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238049, upper bound: 1757.1237472
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239093, upper bound: 1757.1239093
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239093, upper bound: 1757.1239093
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0499375, upper bound: 1757.0499375
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0499375, upper bound: 1757.0499375
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223873, upper bound: 1757.1223873
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223873, upper bound: 1757.1223873
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235577, upper bound: 1757.1235577
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235576, upper bound: 1757.1235577
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875296, upper bound: 1757.0875296
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875296, upper bound: 1757.0875296
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1224207, upper bound: 1757.1224207
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1224207, upper bound: 1757.1224207
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0448863, upper bound: 1757.0448863
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0448863, upper bound: 1757.0448863
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1040101, upper bound: 1757.1034357
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1034357, upper bound: 1757.1034357
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028584, upper bound: 1757.1028584
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028584, upper bound: 1757.1028584
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044451, upper bound: 1757.1043396
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043941, upper bound: 1757.1043396
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0949469, upper bound: 1757.0949469
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0949469, upper bound: 1757.0949469
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1252929, upper bound: 1757.1252607
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1252929, upper bound: 1757.1252607
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1204639, upper bound: 1757.1204639
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1204639, upper bound: 1757.1204639
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0498947, upper bound: 1757.0498947
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0498947, upper bound: 1757.0498947
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076064, upper bound: 1757.1080768
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076064, upper bound: 1757.1080768
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1237515, upper bound: 1757.1237473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1238049, upper bound: 1757.1237472
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1239093, upper bound: 1757.1239093
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1239093, upper bound: 1757.1239093
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0499375, upper bound: 1757.0499375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0499375, upper bound: 1757.0499375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1223873, upper bound: 1757.1223873
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1223873, upper bound: 1757.1223873
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1235577, upper bound: 1757.1235577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1235576, upper bound: 1757.1235577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875091, upper bound: 1757.0875091
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875296, upper bound: 1757.0875296
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875296, upper bound: 1757.0875296
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0875620, upper bound: 1757.0875620
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1224207, upper bound: 1757.1224207
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1224207, upper bound: 1757.1224207
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0448863, upper bound: 1757.0448863
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0448863, upper bound: 1757.0448863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1040101, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1034357, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1028584, upper bound: 1757.1028584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1028584, upper bound: 1757.1028584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1044451, upper bound: 1757.1043396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1043941, upper bound: 1757.1043396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0949469, upper bound: 1757.0949469
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0949469, upper bound: 1757.0949469
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1252929, upper bound: 1757.1252607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1252929, upper bound: 1757.1252607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1204639, upper bound: 1757.1204639
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1204639, upper bound: 1757.1204639
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0498947, upper bound: 1757.0498947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.0498947, upper bound: 1757.0498947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1076064, upper bound: 1757.1080768
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1757.1076064, upper bound: 1757.1080768

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230349, upper bound: 1757.1230322
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230476, upper bound: 1757.1230321
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237908, upper bound: 1757.1237353
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237869, upper bound: 1757.1237353
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1232433, upper bound: 1757.1232433
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1232433, upper bound: 1757.1232433
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1202223, upper bound: 1757.1202223
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1202223, upper bound: 1757.1202223
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0697079, upper bound: 1757.0697079
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0697079, upper bound: 1757.0697079
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217820, upper bound: 1757.1217820
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217820, upper bound: 1757.1217820
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223144, upper bound: 1757.1223144
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223144, upper bound: 1757.1223144
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235427, upper bound: 1757.1235427
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235427, upper bound: 1757.1235427
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235288, upper bound: 1757.1235288
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235288, upper bound: 1757.1235288
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0819169, upper bound: 1757.0819169
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0819169, upper bound: 1757.0819169
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0801154, upper bound: 1757.0801154
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0801154, upper bound: 1757.0801154
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0404483, upper bound: 1757.0404483
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0404483, upper bound: 1757.0404483
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0697762, upper bound: 1757.0697762
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0697762, upper bound: 1757.0697762
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223763
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223762
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1040387, upper bound: 1757.1038757
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1040101, upper bound: 1757.1034357
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1039946, upper bound: 1757.1034357
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1008195, upper bound: 1757.1006005
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1006185, upper bound: 1757.1006005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948456
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948556
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234982, upper bound: 1757.1234982
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235565, upper bound: 1757.1234982
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0976173
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0971673
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1052828
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1049026
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1230349, upper bound: 1757.1230322
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1230476, upper bound: 1757.1230321
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1237908, upper bound: 1757.1237353
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1237869, upper bound: 1757.1237353
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1232433, upper bound: 1757.1232433
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1232433, upper bound: 1757.1232433
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1202223, upper bound: 1757.1202223
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1202223, upper bound: 1757.1202223
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0697079, upper bound: 1757.0697079
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0697079, upper bound: 1757.0697079
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1217820, upper bound: 1757.1217820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1217820, upper bound: 1757.1217820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1223144, upper bound: 1757.1223144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1223144, upper bound: 1757.1223144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235427, upper bound: 1757.1235427
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235427, upper bound: 1757.1235427
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235288, upper bound: 1757.1235288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235288, upper bound: 1757.1235288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0694116, upper bound: 1757.0694116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0819169, upper bound: 1757.0819169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0819169, upper bound: 1757.0819169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0801154, upper bound: 1757.0801154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0801154, upper bound: 1757.0801154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0404483, upper bound: 1757.0404483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0404483, upper bound: 1757.0404483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0697762, upper bound: 1757.0697762
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0697762, upper bound: 1757.0697762
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223762
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1040387, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1040101, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1039946, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1008195, upper bound: 1757.1006005
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1006185, upper bound: 1757.1006005
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948556
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0422157, upper bound: 1757.0422157
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1234982, upper bound: 1757.1234982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1235565, upper bound: 1757.1234982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0976173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0971673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1052828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1049026

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1185087, upper bound: 1757.1185087
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1185087, upper bound: 1757.1185087
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230186, upper bound: 1757.1230186
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1230186, upper bound: 1757.1230186
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237908, upper bound: 1757.1237353
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237352, upper bound: 1757.1237353
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0801496, upper bound: 1757.0801496
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0801496, upper bound: 1757.0801496
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0449366, upper bound: 1757.0449366
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0449366, upper bound: 1757.0449366
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028187, upper bound: 1757.1028187
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028187, upper bound: 1757.1028187
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1200740, upper bound: 1757.1200740
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1200740, upper bound: 1757.1200740
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169167, upper bound: 1757.1169167
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169167, upper bound: 1757.1169167
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0294364, upper bound: 1757.0294364
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0294364, upper bound: 1757.0294364
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217638, upper bound: 1757.1217638
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217638, upper bound: 1757.1217638
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217610, upper bound: 1757.1217610
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1217610, upper bound: 1757.1217610
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015328, upper bound: 1757.1015328
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015328, upper bound: 1757.1015328
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0498233, upper bound: 1757.0498233
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0498233, upper bound: 1757.0498233
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223143, upper bound: 1757.1223144
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223143, upper bound: 1757.1223144
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0939004, upper bound: 1757.0940126
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0939004, upper bound: 1757.0939004
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235331, upper bound: 1757.1235331
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235331, upper bound: 1757.1235331
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1228870, upper bound: 1757.1228870
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1228870, upper bound: 1757.1228870
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1225436, upper bound: 1757.1225437
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1225436, upper bound: 1757.1225437
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226994, upper bound: 1757.1226995
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226994, upper bound: 1757.1226995
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222673
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0792212, upper bound: 1757.0792212
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0792212, upper bound: 1757.0792212
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0816957, upper bound: 1757.0816957
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0816957, upper bound: 1757.0816957
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0638719, upper bound: 1757.0638719
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0638719, upper bound: 1757.0638719
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0818777, upper bound: 1757.0818777
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0818777, upper bound: 1757.0818777
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0632899, upper bound: 1757.0632899
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0632899, upper bound: 1757.0632899
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0766119, upper bound: 1757.0766119
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0766119, upper bound: 1757.0766119
time: 0.71 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1185087, upper bound: 1757.1185087
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1185087, upper bound: 1757.1185087
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1230186, upper bound: 1757.1230186
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1230186, upper bound: 1757.1230186
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1237908, upper bound: 1757.1237353
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1237352, upper bound: 1757.1237353
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0801496, upper bound: 1757.0801496
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0801496, upper bound: 1757.0801496
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0449366, upper bound: 1757.0449366
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0449366, upper bound: 1757.0449366
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1028187, upper bound: 1757.1028187
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1028187, upper bound: 1757.1028187
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1200740, upper bound: 1757.1200740
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1200740, upper bound: 1757.1200740
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1169167, upper bound: 1757.1169167
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1169167, upper bound: 1757.1169167
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1169289, upper bound: 1757.1169289
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0294364, upper bound: 1757.0294364
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0294364, upper bound: 1757.0294364
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1217638, upper bound: 1757.1217638
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1217638, upper bound: 1757.1217638
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1217610, upper bound: 1757.1217610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1217610, upper bound: 1757.1217610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1015328, upper bound: 1757.1015328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1015328, upper bound: 1757.1015328
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1110453, upper bound: 1757.1110453
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1233614, upper bound: 1757.1233614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0498233, upper bound: 1757.0498233
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0498233, upper bound: 1757.0498233
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1223143, upper bound: 1757.1223144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1223143, upper bound: 1757.1223144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0939004, upper bound: 1757.0940126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0939004, upper bound: 1757.0939004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235331, upper bound: 1757.1235331
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235331, upper bound: 1757.1235331
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1235474, upper bound: 1757.1235474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1228870, upper bound: 1757.1228870
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1228870, upper bound: 1757.1228870
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1225436, upper bound: 1757.1225437
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1225436, upper bound: 1757.1225437
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1226994, upper bound: 1757.1226995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1226994, upper bound: 1757.1226995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222673
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.1222646, upper bound: 1757.1222646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0792212, upper bound: 1757.0792212
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0792212, upper bound: 1757.0792212
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872589, upper bound: 1757.0872589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0816957, upper bound: 1757.0816957
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0816957, upper bound: 1757.0816957
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0638719, upper bound: 1757.0638719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0638719, upper bound: 1757.0638719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871852, upper bound: 1757.0871852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0818777, upper bound: 1757.0818777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0818777, upper bound: 1757.0818777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0872279, upper bound: 1757.0872279
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0871902, upper bound: 1757.0871902
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0632899, upper bound: 1757.0632899
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0632899, upper bound: 1757.0632899
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0766119, upper bound: 1757.0766119
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 0, lower bound: -1757.0766119, upper bound: 1757.0766119
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874700, upper bound: 1757.0874700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0874766, upper bound: 1757.0874766
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0875194, upper bound: 1757.0875194
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1223763, upper bound: 1757.1223762
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1199983, upper bound: 1757.1199983
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1040387, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1044527, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1038757, upper bound: 1757.1038757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1040101, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1039946, upper bound: 1757.1034357
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1008195, upper bound: 1757.1006005
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1006185, upper bound: 1757.1006005
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1024127, upper bound: 1757.1024127
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0760619, upper bound: 1757.0760619
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0922106, upper bound: 1757.0922106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0761599, upper bound: 1757.0761599
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1168993, upper bound: 1757.1168993
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1198591, upper bound: 1757.1198591
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1198435, upper bound: 1757.1198435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1195010, upper bound: 1757.1195010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0948456, upper bound: 1757.0948556
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1239503, upper bound: 1757.1239503
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1234982, upper bound: 1757.1234982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1235565, upper bound: 1757.1234982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1204572, upper bound: 1757.1204572
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1197889, upper bound: 1757.1197889
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0976173
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.0971673, upper bound: 1757.0971673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1052828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 0, lower bound: -1757.1048120, upper bound: 1757.1049026
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2002.608154296875
rel_dist={0: [-1757.1270497407063, 1757.1270497407058]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264096, upper bound: 1757.1263577
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263577, upper bound: 1757.1264096
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -1757.1264096, upper bound: 1757.1263577
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -1757.1263577, upper bound: 1757.1264096

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264063, upper bound: 1757.1261136
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263065, upper bound: 1757.1263276
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263458, upper bound: 1757.1264002
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1263458, upper bound: 1757.1264005
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -1757.1264063, upper bound: 1757.1261136
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -1757.1263065, upper bound: 1757.1263276
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -1757.1263458, upper bound: 1757.1264002
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -1757.1263458, upper bound: 1757.1264005

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1264063, upper bound: 1757.1261136
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260734, upper bound: 1757.1258758
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260011, upper bound: 1757.1260930
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258727, upper bound: 1757.1260537
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259296, upper bound: 1757.1260766
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258738, upper bound: 1757.1259912
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258738, upper bound: 1757.1261768
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1264063, upper bound: 1757.1261136
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1260734, upper bound: 1757.1258758
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1260011, upper bound: 1757.1260930
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1258727, upper bound: 1757.1260537
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1259296, upper bound: 1757.1260766
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1258738, upper bound: 1757.1259912
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -1757.1258738, upper bound: 1757.1261768

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261859, upper bound: 1757.1258737
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259995, upper bound: 1757.1258737
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258737, upper bound: 1757.1258737
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259722, upper bound: 1757.1258758
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258790, upper bound: 1757.1260930
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260011, upper bound: 1757.1258737
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243007, upper bound: 1757.1254616
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243007, upper bound: 1757.1243007
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260663
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1260834, upper bound: 1757.1259912
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1259843
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1258561
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1261136, upper bound: 1757.1261136
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1261859, upper bound: 1757.1258737
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1259995, upper bound: 1757.1258737
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1258737, upper bound: 1757.1258737
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1259722, upper bound: 1757.1258758
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1258790, upper bound: 1757.1260930
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1260011, upper bound: 1757.1258737
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1243007, upper bound: 1757.1254616
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1243007, upper bound: 1757.1243007
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260663
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1260834, upper bound: 1757.1259912
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1259843
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1258561

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261077, upper bound: 1757.1261077
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261077, upper bound: 1757.1261077
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1261859, upper bound: 1757.1258737
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259801, upper bound: 1757.1258737
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258737, upper bound: 1757.1258737
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259995, upper bound: 1757.1258737
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053449, upper bound: 1757.1053352
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258762, upper bound: 1757.1258737
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259722, upper bound: 1757.1258758
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258790, upper bound: 1757.1259693
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258788, upper bound: 1757.1260930
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254308
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0423580, upper bound: 1757.0423580
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0423580, upper bound: 1757.0423580
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243739, upper bound: 1757.1242423
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1196262, upper bound: 1757.1196262
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1197876, upper bound: 1757.1199135
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1258945
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1247870
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1249084
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257206
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250059, upper bound: 1757.1255507
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250125, upper bound: 1757.1253125
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1261077, upper bound: 1757.1261077
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1261077, upper bound: 1757.1261077
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1261859, upper bound: 1757.1258737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1259801, upper bound: 1757.1258737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258737, upper bound: 1757.1258737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1259995, upper bound: 1757.1258737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1053449, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258762, upper bound: 1757.1258737
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1259722, upper bound: 1757.1258758
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258790, upper bound: 1757.1259693
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1258788, upper bound: 1757.1260930
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254308
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.0423580, upper bound: 1757.0423580
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.0423580, upper bound: 1757.0423580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1243739, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1196262, upper bound: 1757.1196262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1197876, upper bound: 1757.1199135
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1258945
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1247870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1249084
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257206
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1250059, upper bound: 1757.1255507
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1757.1250125, upper bound: 1757.1253125

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0989384, upper bound: 1757.0989384
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0989384, upper bound: 1757.0989384
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246396, upper bound: 1757.1244783
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246757, upper bound: 1757.1244783
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184938, upper bound: 1757.1184135
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184938, upper bound: 1757.1184135
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050505, upper bound: 1757.1050415
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258429, upper bound: 1757.1258415
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1259632, upper bound: 1757.1258659
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239005, upper bound: 1757.1239005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239005, upper bound: 1757.1239012
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243720, upper bound: 1757.1242423
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0499215, upper bound: 1757.0499215
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0499215, upper bound: 1757.0499215
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1193693, upper bound: 1757.1197155
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1192947, upper bound: 1757.1192947
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1256766
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256477, upper bound: 1757.1256477
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256477, upper bound: 1757.1258768
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1247870
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1244671
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244647, upper bound: 1757.1248997
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1245094
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1155347, upper bound: 1757.1155347
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1155347, upper bound: 1757.1156150
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166339
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1256395
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1253708
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1254356
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1255479
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1253388
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1183596, upper bound: 1757.1187478
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1183596, upper bound: 1757.1187478
time: 0.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0989384, upper bound: 1757.0989384
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0989384, upper bound: 1757.0989384
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258507, upper bound: 1757.1258507
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1246396, upper bound: 1757.1244783
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1246757, upper bound: 1757.1244783
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1184938, upper bound: 1757.1184135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1184938, upper bound: 1757.1184135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1050505, upper bound: 1757.1050415
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258429, upper bound: 1757.1258415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1259632, upper bound: 1757.1258659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1239005, upper bound: 1757.1239005
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1239005, upper bound: 1757.1239012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1254306, upper bound: 1757.1254306
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1243720, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0499215, upper bound: 1757.0499215
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.0499215, upper bound: 1757.0499215
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1193693, upper bound: 1757.1197155
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1192947, upper bound: 1757.1192947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1260766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256766, upper bound: 1757.1256766
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256477, upper bound: 1757.1256477
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1256477, upper bound: 1757.1258768
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1247870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1244671, upper bound: 1757.1244671
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1244647, upper bound: 1757.1248997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1245094
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1155347, upper bound: 1757.1155347
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1155347, upper bound: 1757.1156150
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166386
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166339
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1256395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1253708
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1257978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1254356, upper bound: 1757.1254356
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1255479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1253388
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1183596, upper bound: 1757.1187478
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 0, lower bound: -1757.1183596, upper bound: 1757.1187478

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0845274, upper bound: 1757.0845274
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0845274, upper bound: 1757.0845274
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0877567, upper bound: 1757.0877567
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0877567, upper bound: 1757.0877567
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237226, upper bound: 1757.1237226
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237226, upper bound: 1757.1237226
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255623, upper bound: 1757.1255623
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255623, upper bound: 1757.1255623
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050005, upper bound: 1757.1049898
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050000, upper bound: 1757.1049898
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235655, upper bound: 1757.1235150
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236946, upper bound: 1757.1235150
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112439, upper bound: 1757.1112439
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112439, upper bound: 1757.1112439
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184885, upper bound: 1757.1184135
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184135, upper bound: 1757.1184135
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184885, upper bound: 1757.1184135
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184738, upper bound: 1757.1184135
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0921799, upper bound: 1757.0921799
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0921799, upper bound: 1757.0921799
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0923078, upper bound: 1757.0923078
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0923078, upper bound: 1757.0923078
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236424, upper bound: 1757.1236424
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236424, upper bound: 1757.1236424
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258429, upper bound: 1757.1258415
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258659
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255370, upper bound: 1757.1255370
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255370, upper bound: 1757.1255370
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256937
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231444, upper bound: 1757.1231444
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231444, upper bound: 1757.1231444
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1228850, upper bound: 1757.1228850
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1228850, upper bound: 1757.1228850
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0978523, upper bound: 1757.0978523
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0978523, upper bound: 1757.0978523
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236180, upper bound: 1757.1236180
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236180, upper bound: 1757.1236180
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254202, upper bound: 1757.1254202
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1254202, upper bound: 1757.1254202
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237825, upper bound: 1757.1237825
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237825, upper bound: 1757.1237825
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1124547, upper bound: 1757.1124547
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1124547, upper bound: 1757.1124547
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243720, upper bound: 1757.1242423
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045421, upper bound: 1757.1045421
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045421, upper bound: 1757.1045421
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184845, upper bound: 1757.1184845
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1184845, upper bound: 1757.1184845
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189154, upper bound: 1757.1190666
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1189154, upper bound: 1757.1190666
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246836, upper bound: 1757.1246836
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246836, upper bound: 1757.1246836
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038111, upper bound: 1757.1038111
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1038111, upper bound: 1757.1038111
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255985, upper bound: 1757.1258483
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1255985, upper bound: 1757.1255985
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1116806, upper bound: 1757.1116806
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1116806, upper bound: 1757.1116806
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1079623, upper bound: 1757.1079623
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1079623, upper bound: 1757.1079623
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244549, upper bound: 1757.1248997
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1244203
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246390, upper bound: 1757.1244229
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1245094
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1154577, upper bound: 1757.1154577
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1154577, upper bound: 1757.1154577
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076020, upper bound: 1757.1076020
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076020, upper bound: 1757.1076020
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1157418, upper bound: 1757.1159036
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1157418, upper bound: 1757.1157418
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1164766
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166339
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1253708
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1256395
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253555, upper bound: 1757.1253555
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1253555, upper bound: 1757.1253555
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1021661, upper bound: 1757.1021661
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1021661, upper bound: 1757.1021661
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244380, upper bound: 1757.1244380
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244380, upper bound: 1757.1244380
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1252691
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1255479
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1182619
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1182619
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1178893
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1179226
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020238, upper bound: 1757.1021147
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020238, upper bound: 1757.1021194
time: 0.71 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0845274, upper bound: 1757.0845274
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0845274, upper bound: 1757.0845274
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0877567, upper bound: 1757.0877567
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0877567, upper bound: 1757.0877567
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237226, upper bound: 1757.1237226
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237226, upper bound: 1757.1237226
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239746, upper bound: 1757.1239746
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255623, upper bound: 1757.1255623
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255623, upper bound: 1757.1255623
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258392, upper bound: 1757.1258392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050005, upper bound: 1757.1049898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050000, upper bound: 1757.1049898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1235655, upper bound: 1757.1235150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1236946, upper bound: 1757.1235150
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1112439, upper bound: 1757.1112439
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1112439, upper bound: 1757.1112439
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184885, upper bound: 1757.1184135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184135, upper bound: 1757.1184135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184885, upper bound: 1757.1184135
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184738, upper bound: 1757.1184135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050415
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1050415, upper bound: 1757.1050414
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0921799, upper bound: 1757.0921799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0921799, upper bound: 1757.0921799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0923078, upper bound: 1757.0923078
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0923078, upper bound: 1757.0923078
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1053352, upper bound: 1757.1053352
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1236424, upper bound: 1757.1236424
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1236424, upper bound: 1757.1236424
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258429, upper bound: 1757.1258415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258415, upper bound: 1757.1258415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1258628, upper bound: 1757.1258628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255370, upper bound: 1757.1255370
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255370, upper bound: 1757.1255370
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256937
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1256937, upper bound: 1757.1256997
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1231444, upper bound: 1757.1231444
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1231444, upper bound: 1757.1231444
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1228850, upper bound: 1757.1228850
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1228850, upper bound: 1757.1228850
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0978523, upper bound: 1757.0978523
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0978523, upper bound: 1757.0978523
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.0979648, upper bound: 1757.0979648
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1236180, upper bound: 1757.1236180
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1236180, upper bound: 1757.1236180
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1254202, upper bound: 1757.1254202
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1254202, upper bound: 1757.1254202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237825, upper bound: 1757.1237825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237825, upper bound: 1757.1237825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1237906, upper bound: 1757.1237906
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1124547, upper bound: 1757.1124547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1124547, upper bound: 1757.1124547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1242423, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1243720, upper bound: 1757.1242423
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1045421, upper bound: 1757.1045421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1045421, upper bound: 1757.1045421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184845, upper bound: 1757.1184845
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1184845, upper bound: 1757.1184845
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1189154, upper bound: 1757.1190666
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1189154, upper bound: 1757.1190666
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1246836, upper bound: 1757.1246836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1246836, upper bound: 1757.1246836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1038111, upper bound: 1757.1038111
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1038111, upper bound: 1757.1038111
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255985, upper bound: 1757.1258483
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1255985, upper bound: 1757.1255985
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1116806, upper bound: 1757.1116806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1116806, upper bound: 1757.1116806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1079623, upper bound: 1757.1079623
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1079623, upper bound: 1757.1079623
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244549, upper bound: 1757.1248997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1244203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1246390, upper bound: 1757.1244229
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244203, upper bound: 1757.1245094
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1154577, upper bound: 1757.1154577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1154577, upper bound: 1757.1154577
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1076020, upper bound: 1757.1076020
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1076020, upper bound: 1757.1076020
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1157418, upper bound: 1757.1159036
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1157418, upper bound: 1757.1157418
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1164766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1164766, upper bound: 1757.1166339
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1253708
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1253708, upper bound: 1757.1256395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1253555, upper bound: 1757.1253555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1253555, upper bound: 1757.1253555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1021661, upper bound: 1757.1021661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1021661, upper bound: 1757.1021661
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244380, upper bound: 1757.1244380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1244380, upper bound: 1757.1244380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1252691
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1249548, upper bound: 1757.1255479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1182619
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1182619
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1178893
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1177384, upper bound: 1757.1179226
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1020238, upper bound: 1757.1021147
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.85
Output dim: 0, lower bound: -1757.1020238, upper bound: 1757.1021194

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0838183, upper bound: 1757.0838183
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0838183, upper bound: 1757.0838183
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0782159, upper bound: 1757.0782159
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0782159, upper bound: 1757.0782159
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0689980, upper bound: 1757.0689980
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0689980, upper bound: 1757.0689980
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0338443, upper bound: 1757.0338443
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0338443, upper bound: 1757.0338443
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1209669, upper bound: 1757.1209669
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1209669, upper bound: 1757.1209669
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234813, upper bound: 1757.1234813
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234813, upper bound: 1757.1234813
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237064, upper bound: 1757.1237064
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1237064, upper bound: 1757.1237064
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226730, upper bound: 1757.1226730
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1226730, upper bound: 1757.1226730
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238745, upper bound: 1757.1238745
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238745, upper bound: 1757.1238745
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238745, upper bound: 1757.1238745
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238745, upper bound: 1757.1238745
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0889281, upper bound: 1757.0889281
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0889281, upper bound: 1757.0889281
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239714, upper bound: 1757.1239714
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234595, upper bound: 1757.1234595
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234595, upper bound: 1757.1234595
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233922, upper bound: 1757.1233922
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233922, upper bound: 1757.1233921
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251743, upper bound: 1757.1251743
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251743, upper bound: 1757.1251743
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239583, upper bound: 1757.1239583
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1239583, upper bound: 1757.1239583
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1050005, upper bound: 1757.1049898
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1049997, upper bound: 1757.1049898
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1035589, upper bound: 1757.1035589
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1035589, upper bound: 1757.1035589
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0793616, upper bound: 1757.0793616
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0793616, upper bound: 1757.0793616
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014056, upper bound: 1757.1014056
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1014056, upper bound: 1757.1014056
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227654, upper bound: 1757.1227113
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227113, upper bound: 1757.1227113
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236946, upper bound: 1757.1235150
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1235515, upper bound: 1757.1235150
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0865446, upper bound: 1757.0865446
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0865446, upper bound: 1757.0865446
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066643, upper bound: 1757.1066643
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1066643, upper bound: 1757.1066643
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234999, upper bound: 1757.1234999
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234999, upper bound: 1757.1234999
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0935781, upper bound: 1757.0935781
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0935781, upper bound: 1757.0935781
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244366, upper bound: 1757.1244366
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2002.608154296875
rel_dist={0: [-1757.1264512206008, 1757.1264512206008]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250983, upper bound: 1757.1250984
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250984, upper bound: 1757.1250983
time: 0.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -1757.1250983, upper bound: 1757.1250984
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -1757.1250984, upper bound: 1757.1250983

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0501646, upper bound: 1757.0500331
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0501646, upper bound: 1757.0500331
time: 0.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250984, upper bound: 1757.1250073
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250915, upper bound: 1757.1250983
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 0, lower bound: -1757.0501646, upper bound: 1757.0500331
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 0, lower bound: -1757.0501646, upper bound: 1757.0500331
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -1757.1250984, upper bound: 1757.1250073
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -1757.1250915, upper bound: 1757.1250983

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1248144, upper bound: 1757.1248144
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1248284, upper bound: 1757.1250073
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1247879, upper bound: 1757.1250780
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1250607, upper bound: 1757.1248578
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1757.1248144, upper bound: 1757.1248144
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1757.1248284, upper bound: 1757.1250073
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1757.1247879, upper bound: 1757.1250780
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1757.1250607, upper bound: 1757.1248578

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246021, upper bound: 1757.1244257
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1246021, upper bound: 1757.1244257
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1248054, upper bound: 1757.1249110
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1248054, upper bound: 1757.1249981
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1247349
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246217
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.60 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1246021, upper bound: 1757.1244257
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1246021, upper bound: 1757.1244257
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1248054, upper bound: 1757.1249110
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1248054, upper bound: 1757.1249981
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1247349
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246217
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.60
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244257, upper bound: 1757.1244257
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244257, upper bound: 1757.1244257
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244512, upper bound: 1757.1244242
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1247228, upper bound: 1757.1244242
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243602, upper bound: 1757.1243602
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243602, upper bound: 1757.1243602
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244158, upper bound: 1757.1245810
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244158, upper bound: 1757.1245489
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246590
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1182070
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1182070
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220293
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
time: 0.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.60 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1244257, upper bound: 1757.1244257
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1244257, upper bound: 1757.1244257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1244512, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1247228, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1243602, upper bound: 1757.1243602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1243602, upper bound: 1757.1243602
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1244158, upper bound: 1757.1245810
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1244158, upper bound: 1757.1245489
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246590
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1182070
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1182070
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220293
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.60
Output dim: 0, lower bound: -1757.1231271, upper bound: 1757.1231271

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181034, upper bound: 1757.1181282
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181874, upper bound: 1757.1181282
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1241718, upper bound: 1757.1241718
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1241718, upper bound: 1757.1241718
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1247153, upper bound: 1757.1244242
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1016473
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015624
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1221285, upper bound: 1757.1221285
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1221285, upper bound: 1757.1221285
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238650, upper bound: 1757.1238650
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1238650, upper bound: 1757.1238650
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181218, upper bound: 1757.1182717
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181218, upper bound: 1757.1182601
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1244157
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244159, upper bound: 1757.1245489
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1246261
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1246497
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1245114
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1180021
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1181498
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175126, upper bound: 1757.1179011
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175126, upper bound: 1757.1175126
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220293
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213839, upper bound: 1757.1213839
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213839, upper bound: 1757.1213839
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.67 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1181034, upper bound: 1757.1181282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1181874, upper bound: 1757.1181282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1241718, upper bound: 1757.1241718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1241718, upper bound: 1757.1241718
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1247153, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1016473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015624
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1221285, upper bound: 1757.1221285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1221285, upper bound: 1757.1221285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1238650, upper bound: 1757.1238650
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1238650, upper bound: 1757.1238650
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1181218, upper bound: 1757.1182717
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1181218, upper bound: 1757.1182601
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1244157
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1244159, upper bound: 1757.1245489
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1246261
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1246497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1245114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1180021
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1180021, upper bound: 1757.1181498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1175126, upper bound: 1757.1179011
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1175126, upper bound: 1757.1175126
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220293
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1213839, upper bound: 1757.1213839
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1213839, upper bound: 1757.1213839
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1084352, upper bound: 1757.1084723
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1084352, upper bound: 1757.1084723
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1178588, upper bound: 1757.1177998
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177998, upper bound: 1757.1178270
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1247153, upper bound: 1757.1244242
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1016337
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1016408
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015624
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015524
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1178139, upper bound: 1757.1178139
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1178139, upper bound: 1757.1179812
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181172, upper bound: 1757.1181284
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1181172, upper bound: 1757.1182600
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1225141, upper bound: 1757.1225141
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1225141, upper bound: 1757.1225141
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1244390
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1245489
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1229709
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1230288
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1246497
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1243751
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1245114
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1243861
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246772
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1150854, upper bound: 1757.1150854
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1150854, upper bound: 1757.1150905
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179765, upper bound: 1757.1179897
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1179765, upper bound: 1757.1181498
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1140982, upper bound: 1757.1141045
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1140982, upper bound: 1757.1141045
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175050, upper bound: 1757.1175050
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175050, upper bound: 1757.1175050
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1209880, upper bound: 1757.1209880
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1209880, upper bound: 1757.1209918
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213707, upper bound: 1757.1213707
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213707, upper bound: 1757.1213706
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1211743, upper bound: 1757.1211743
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1211743, upper bound: 1757.1211743
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
time: 0.77 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.25 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1084352, upper bound: 1757.1084723
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1084352, upper bound: 1757.1084723
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1178588, upper bound: 1757.1177998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1177998, upper bound: 1757.1178270
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1247153, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1244242, upper bound: 1757.1244242
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1016337
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1016408
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015624
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015524
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1178139, upper bound: 1757.1178139
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1178139, upper bound: 1757.1179812
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1181172, upper bound: 1757.1181284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1181172, upper bound: 1757.1182600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1225141, upper bound: 1757.1225141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1225141, upper bound: 1757.1225141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1244390
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1244157, upper bound: 1757.1245489
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1229709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1230288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1246497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1243751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1245114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1243861
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246772
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1246924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1150854, upper bound: 1757.1150854
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1150854, upper bound: 1757.1150905
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1179765, upper bound: 1757.1179897
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1179765, upper bound: 1757.1181498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1140982, upper bound: 1757.1141045
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1140982, upper bound: 1757.1141045
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1175050, upper bound: 1757.1175050
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1175050, upper bound: 1757.1175050
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1220223, upper bound: 1757.1220223
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1209880, upper bound: 1757.1209880
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1209880, upper bound: 1757.1209918
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1213707, upper bound: 1757.1213707
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1213707, upper bound: 1757.1213706
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1211743, upper bound: 1757.1211743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1211743, upper bound: 1757.1211743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1231166, upper bound: 1757.1231166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -1757.1223955, upper bound: 1757.1223955

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022302, upper bound: 1757.1022302
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022302, upper bound: 1757.1022422
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022302, upper bound: 1757.1022302
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022302, upper bound: 1757.1022422
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1176597, upper bound: 1757.1176597
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1176597, upper bound: 1757.1176597
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177998, upper bound: 1757.1178270
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1177998, upper bound: 1757.1177998
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1212591, upper bound: 1757.1212591
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1212591, upper bound: 1757.1212591
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1222030, upper bound: 1757.1222030
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234120, upper bound: 1757.1234120
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1243861
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243861, upper bound: 1757.1243861
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1013336, upper bound: 1757.1013947
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1013336, upper bound: 1757.1013336
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1051073, upper bound: 1757.1051073
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1051073, upper bound: 1757.1051073
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244225, upper bound: 1757.1244225
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244225, upper bound: 1757.1244225
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1016218
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015492, upper bound: 1757.1015492
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0749898, upper bound: 1757.0749898
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0749898, upper bound: 1757.0749898
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012108, upper bound: 1757.1012108
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012108, upper bound: 1757.1012342
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015524
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1015524, upper bound: 1757.1015524
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1219158, upper bound: 1757.1219158
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213247, upper bound: 1757.1213247
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1213247, upper bound: 1757.1213247
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1218020, upper bound: 1757.1218020
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1211717, upper bound: 1757.1211717
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1211717, upper bound: 1757.1211717
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236452, upper bound: 1757.1236452
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236452, upper bound: 1757.1236452
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236576, upper bound: 1757.1236576
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236538, upper bound: 1757.1236538
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236538, upper bound: 1757.1236538
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233872, upper bound: 1757.1233872
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1233872, upper bound: 1757.1233872
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149961, upper bound: 1757.1149961
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1149961, upper bound: 1757.1149961
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1114442, upper bound: 1757.1114708
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1114442, upper bound: 1757.1114727
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1150550, upper bound: 1757.1150550
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1150550, upper bound: 1757.1150550
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1151470, upper bound: 1757.1153091
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1151470, upper bound: 1757.1152044
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0353738, upper bound: 1757.0353738
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0353738, upper bound: 1757.0353738
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1190193, upper bound: 1757.1190193
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1190193, upper bound: 1757.1190193
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234367, upper bound: 1757.1234516
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1234367, upper bound: 1757.1234392
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244154, upper bound: 1757.1245131
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1244154, upper bound: 1757.1244154
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1229709
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1227162
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1227162
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1227162, upper bound: 1757.1230288
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1243747
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1246497
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1243747
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243747, upper bound: 1757.1243751
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1243754
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1243754, upper bound: 1757.1245022
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236640, upper bound: 1757.1236640
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1236640, upper bound: 1757.1236640
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 1.41 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2002.608154296875
rel_dist={0: [-1757.1254498169274, 1757.1254498169274]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1132.97 seconds
