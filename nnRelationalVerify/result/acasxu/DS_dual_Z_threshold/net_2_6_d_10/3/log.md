## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 23.9931544845


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315)
1: (-1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685)
2: (-1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412)
3: (-1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150)
4: (-1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.19 + 1.27 = 3.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1137231

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958
time: 0.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.99 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0194971
time: 0.37 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0194971, upper bound: 24.0191958
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0194971, upper bound: 24.0191958
time: 0.40 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0194971
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -24.0194971, upper bound: 24.0191958
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -24.0194971, upper bound: 24.0191958

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0205856
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0201756
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0194753
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0193389
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0205856, upper bound: 24.0191743
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.17 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0205856
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0201756
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0194753
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0193389
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -24.0205856, upper bound: 24.0191743

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0205856
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0201756
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0194753
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0193389
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0205856, upper bound: 24.0191743
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0205856
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0201756
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0194753
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0193389
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0205856, upper bound: 24.0191743
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0198819
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0185370
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180081
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176317
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0180081, upper bound: 24.0176202
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0185370, upper bound: 24.0176202
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0198819
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0185370
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176317
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0180081, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0185370, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9955451
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9955010
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9963902, upper bound: 23.9954191
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9974456, upper bound: 23.9954191
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9963902, upper bound: 23.9954191
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9974456, upper bound: 23.9954191
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974456
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9963902
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974456
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9963902
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9971113, upper bound: 23.9954191
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9974526, upper bound: 23.9954191
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954275, upper bound: 23.9954191
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9955451, upper bound: 23.9954191
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.39 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9955451
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9955010
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9963902, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9974456, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9963902, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9974456, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954275
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974456
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9963902
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974456
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9963902
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9971113, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9974526, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954275, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9955451, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845186
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845185
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845185
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845185
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845183
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845195
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845186
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845195
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845189
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845195
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845196
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845190, upper bound: 23.9845182
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9846839, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845190, upper bound: 23.9845182
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845186
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845183
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845186
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845195
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845183
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845190
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845190
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845187, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845195, upper bound: 23.9845182
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9846839, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845187, upper bound: 23.9845182
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845195, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9846839, upper bound: 23.9845182
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 2.42 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.46 + 418.19 = 421.65 seconds
