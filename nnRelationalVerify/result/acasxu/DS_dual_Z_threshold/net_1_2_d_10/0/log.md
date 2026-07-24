## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1757.072574941339


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543)
1: (-485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883)
2: (-554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562)
3: (-788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867)
4: (-931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 1.99 = 2.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1251902
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1251902
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -1757.1251902, upper bound: 1757.1252887

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044197, upper bound: 1757.1044155
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044197, upper bound: 1757.1044208
time: 0.64 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044208, upper bound: 1757.1044197
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044155, upper bound: 1757.1044230
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -1757.1044197, upper bound: 1757.1044155
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -1757.1044197, upper bound: 1757.1044208
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -1757.1044208, upper bound: 1757.1044197
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -1757.1044155, upper bound: 1757.1044230

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042612, upper bound: 1757.1043060
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042612, upper bound: 1757.1042610
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042612
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042612
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1043049
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.39 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042612, upper bound: 1757.1043060
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042612, upper bound: 1757.1042610
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042612
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042612
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1043049
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1757.1042610, upper bound: 1757.1042610

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543
1: -485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883
2: -554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562
3: -788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867
4: -931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1757.0335472, upper bound: 1757.0335472

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.79 + 30.48 = 33.27 seconds
