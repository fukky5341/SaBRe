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
execution time: IAR + RelationalAnalysis = 1.31 + 2.18 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -77.9535863, upper bound: 77.9535863

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9526108, upper bound: 77.9526590
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9526108, upper bound: 77.9526108
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.72 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 3, lower bound: -77.9526108, upper bound: 77.9526590
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 3, lower bound: -77.9526108, upper bound: 77.9526108

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458250, upper bound: 77.9459570
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458250, upper bound: 77.9459570
time: 0.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 3, lower bound: -77.9458250, upper bound: 77.9459570
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 3, lower bound: -77.9458250, upper bound: 77.9459570
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 3, lower bound: -77.9459570, upper bound: 77.9458250

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9458974
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9459340
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9458974
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9459340
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458974, upper bound: 77.9458006
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9458974, upper bound: 77.9458006
time: 1.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.94 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9458974
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9459340
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9458974
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458006, upper bound: 77.9459340
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458974, upper bound: 77.9458006
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9459340, upper bound: 77.9458006
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -77.9458974, upper bound: 77.9458006

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588
time: 0.76 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588
time: 0.74 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588
time: 0.70 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.01
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173
1: -32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176
2: -28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595
3: -39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838
4: -36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
time: 0.67 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384780
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.14
Output dim: 3, lower bound: -77.9382667, upper bound: 77.9382667
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384922
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384812
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384947, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384812, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384922, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384927, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384588, upper bound: 77.9384588
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.14
Output dim: 3, lower bound: -77.9384780, upper bound: 77.9384588

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.49 + 416.68 = 420.17 seconds
