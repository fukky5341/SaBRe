## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 77.93799558274


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173)
1: (-32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176)
2: (-28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595)
3: (-39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838)
4: (-36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.09 + 2.14 = 3.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -77.9535863, upper bound: 77.9535863

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9527142, upper bound: 77.9528348
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9528348, upper bound: 77.9527142
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.50 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -77.9527142, upper bound: 77.9528348
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -77.9528348, upper bound: 77.9527142

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493493, upper bound: 77.9497209
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493493, upper bound: 77.9497209
time: 0.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525558, upper bound: 77.9525558
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9525558, upper bound: 77.9527103
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.83 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 3, lower bound: -77.9493493, upper bound: 77.9497209
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 3, lower bound: -77.9493493, upper bound: 77.9497209
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 3, lower bound: -77.9525558, upper bound: 77.9525558
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 3, lower bound: -77.9525558, upper bound: 77.9527103

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493411, upper bound: 77.9493644
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493411, upper bound: 77.9497130
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489558, upper bound: 77.9492085
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489558, upper bound: 77.9489558
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9512033, upper bound: 77.9512033
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9512033, upper bound: 77.9512033
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9520381, upper bound: 77.9526903
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9520381, upper bound: 77.9520381
time: 1.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.68 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9493411, upper bound: 77.9493644
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9493411, upper bound: 77.9497130
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9489558, upper bound: 77.9492085
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9489558, upper bound: 77.9489558
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9512033, upper bound: 77.9512033
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9512033, upper bound: 77.9512033
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9520381, upper bound: 77.9526903
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 3, lower bound: -77.9520381, upper bound: 77.9520381

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9491082, upper bound: 77.9491206
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9491082, upper bound: 77.9491082
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9277007, upper bound: 77.9277007
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9277007, upper bound: 77.9277007
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090755, upper bound: 77.9090755
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090755, upper bound: 77.9090755
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9507877, upper bound: 77.9512033
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9507877, upper bound: 77.9505844
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9509737, upper bound: 77.9509737
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9509737, upper bound: 77.9509970
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9461864, upper bound: 77.9461864
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9461864, upper bound: 77.9461864
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493251, upper bound: 77.9493100
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9493251, upper bound: 77.9493100
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9491082, upper bound: 77.9491206
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9491082, upper bound: 77.9491082
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9277007, upper bound: 77.9277007
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9277007, upper bound: 77.9277007
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9090755, upper bound: 77.9090755
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9090755, upper bound: 77.9090755
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9507877, upper bound: 77.9512033
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9507877, upper bound: 77.9505844
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9509737, upper bound: 77.9509737
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9509737, upper bound: 77.9509970
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9461864, upper bound: 77.9461864
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9461864, upper bound: 77.9461864
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9493251, upper bound: 77.9493100
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 3, lower bound: -77.9493251, upper bound: 77.9493100

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9394482, upper bound: 77.9394482
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9394482, upper bound: 77.9394482
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9490930, upper bound: 77.9490930
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9490930, upper bound: 77.9490930
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9506132, upper bound: 77.9509178
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9511794, upper bound: 77.9503404
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9491966, upper bound: 77.9489915
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459872, upper bound: 77.9459872
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459872, upper bound: 77.9459872
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9394482, upper bound: 77.9394482
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9394482, upper bound: 77.9394482
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9490930, upper bound: 77.9490930
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9490930, upper bound: 77.9490930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9506132, upper bound: 77.9509178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9511794, upper bound: 77.9503404
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9489915, upper bound: 77.9489915
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9491966, upper bound: 77.9489915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9459872, upper bound: 77.9459872
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9459872, upper bound: 77.9459872
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386692, upper bound: 77.9386691
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386693, upper bound: 77.9386691
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489057, upper bound: 77.9488759
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9488759, upper bound: 77.9488759
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9505060, upper bound: 77.9509178
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9505990, upper bound: 77.9503208
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489603, upper bound: 77.9487938
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9489603, upper bound: 77.9487938
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9487670, upper bound: 77.9487670
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9490281, upper bound: 77.9487669
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9380326, upper bound: 77.9380326
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9380326, upper bound: 77.9380326
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.70 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9386692, upper bound: 77.9386691
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9386693, upper bound: 77.9386691
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9489057, upper bound: 77.9488759
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9488759, upper bound: 77.9488759
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9505060, upper bound: 77.9509178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9505990, upper bound: 77.9503208
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9489603, upper bound: 77.9487938
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9489603, upper bound: 77.9487938
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9487670, upper bound: 77.9487670
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9490281, upper bound: 77.9487669
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9380326, upper bound: 77.9380326
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9380326, upper bound: 77.9380326
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9386805, upper bound: 77.9386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9487453, upper bound: 77.9487453
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9487462, upper bound: 77.9487452
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9460412, upper bound: 77.9459481
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9460412, upper bound: 77.9459481
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 0.95 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.84 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9487453, upper bound: 77.9487453
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9487462, upper bound: 77.9487452
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9460412, upper bound: 77.9459481
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9460412, upper bound: 77.9459481
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9372938, upper bound: 77.9372938
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9267869, upper bound: 77.9267869
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9459481, upper bound: 77.9459481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9446385, upper bound: 77.9446385
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9447646, upper bound: 77.9446385
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393170, upper bound: 77.9392462
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393304, upper bound: 77.9392462
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.10 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.85 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9090446, upper bound: 77.9090446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9276650, upper bound: 77.9276650
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9446385, upper bound: 77.9446385
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9447646, upper bound: 77.9446385
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9393170, upper bound: 77.9392462
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9393304, upper bound: 77.9392462
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9392462, upper bound: 77.9392462
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9385033, upper bound: 77.9385033
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385101, upper bound: 77.9384763
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
time: 0.62 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.39 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9378692, upper bound: 77.9378692
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9275705, upper bound: 77.9275705
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9385101, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9266930, upper bound: 77.9266930
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.39
Output dim: 3, lower bound: -77.9384763, upper bound: 77.9384763

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371277, upper bound: 77.9371016
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
time: 0.69 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.89 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371277, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9371016, upper bound: 77.9371016
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.89
Output dim: 3, lower bound: -77.9089635, upper bound: 77.9089635

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.23 + 281.33 = 284.56 seconds
