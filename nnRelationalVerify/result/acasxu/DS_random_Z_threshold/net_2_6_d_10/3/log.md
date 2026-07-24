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
execution time: IAR + RelationalAnalysis = 0.78 + 1.16 = 1.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1137231

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1018562
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.1018562, upper bound: 24.1137231
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1018562
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -24.1018562, upper bound: 24.1137231

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.1103237, upper bound: 24.1018414
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.1137068, upper bound: 24.0985470
time: 0.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0946863, upper bound: 24.1081187
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0947773, upper bound: 24.1074547
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -24.1103237, upper bound: 24.1018414
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -24.1137068, upper bound: 24.0985470
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -24.0946863, upper bound: 24.1081187
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -24.0947773, upper bound: 24.1074547

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0790397, upper bound: 24.0787732
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0792683, upper bound: 24.0787690
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0746285, upper bound: 24.0754980
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0748340, upper bound: 24.0751302
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0754396, upper bound: 24.0747619
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0754771, upper bound: 24.0746285
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.33 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0790397, upper bound: 24.0787732
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0792683, upper bound: 24.0787690
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0746285, upper bound: 24.0754980
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0748340, upper bound: 24.0751302
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0754396, upper bound: 24.0747619
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.33
Output dim: 0, lower bound: -24.0754771, upper bound: 24.0746285

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0748249
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0748285, upper bound: 24.0744192
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0754839
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0748285
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0182718
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0178256
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0199485
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0180558
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0748253, upper bound: 24.0746146
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0748253, upper bound: 24.0744192
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0193389, upper bound: 24.0191743
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0201756, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0191743, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0194753, upper bound: 24.0191743
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0748249
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0748285, upper bound: 24.0744192
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0754839
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0744192, upper bound: 24.0748285
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0182718
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0178256
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0199485
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0176396, upper bound: 24.0180558
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0748253, upper bound: 24.0746146
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.34
Output dim: 0, lower bound: -24.0748253, upper bound: 24.0744192

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176317, upper bound: 24.0176202
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0185370, upper bound: 24.0176202
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0198819
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180081
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0185370
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176317
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
time: 0.26 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0178054, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0182521, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0199287, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176317, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0185370, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0198819
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180081
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0185370
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0199287
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0180359
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176317
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0176202
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0182521
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.33
Output dim: 0, lower bound: -24.0176202, upper bound: 24.0178054

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055836, upper bound: 24.0055827
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055834
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055829
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0064101, upper bound: 24.0055829
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0030012, upper bound: 24.0026606
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0079834, upper bound: 24.0055827
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910443, upper bound: 23.9910422
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055834, upper bound: 24.0055827
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0053073
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0047452
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9912529
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0063261
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0074638
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0027711
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910443
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910482
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
time: 0.26 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055836, upper bound: 24.0055827
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055834
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055829
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0064101, upper bound: 24.0055829
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0030012, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0079834, upper bound: 24.0055827
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910443, upper bound: 23.9910422
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055834, upper bound: 24.0055827
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0053073
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0047452
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9912529
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0063261
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0074638
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9974526
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9971113
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0026606
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0026606, upper bound: 24.0027711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910422
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -24.0055827, upper bound: 24.0055827
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910443
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9910422, upper bound: 23.9910482
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -23.9954191, upper bound: 23.9954191

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9732222, upper bound: 23.9731982
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9779388, upper bound: 23.9777911
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845190, upper bound: 23.9845182
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9846839, upper bound: 23.9845182
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9782490, upper bound: 23.9777136
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9782803, upper bound: 23.9777136
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9732254, upper bound: 23.9731947
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9732241, upper bound: 23.9731947
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9754138, upper bound: 23.9753980
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9913262, upper bound: 23.9888674
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9899396, upper bound: 23.9888674
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845183, upper bound: 23.9845182
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845185, upper bound: 23.9845182
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9754622
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9756357
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9908606
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9613475
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607531
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9608036
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607668
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9732241
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9604693
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9604693
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 4

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845185
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9608036
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607893
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9894609
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315
1: -1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685
2: -1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412
3: -1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150
4: -1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 4
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9732222, upper bound: 23.9731982
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9779388, upper bound: 23.9777911
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845190, upper bound: 23.9845182
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9846839, upper bound: 23.9845182
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9782490, upper bound: 23.9777136
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9782803, upper bound: 23.9777136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9732254, upper bound: 23.9731947
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9732241, upper bound: 23.9731947
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9754138, upper bound: 23.9753980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9913262, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9899396, upper bound: 23.9888674
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845183, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9753980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845185, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9754622
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9753980, upper bound: 23.9756357
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9908606
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9613475
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607531
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9608036
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607668
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9777136, upper bound: 23.9777136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9732241
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9604693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9604693
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845185
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9846839
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9608036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9604693, upper bound: 23.9607893
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9731947, upper bound: 23.9731947
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9894609
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9888674, upper bound: 23.9888674
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.57
Output dim: 0, lower bound: -23.9845182, upper bound: 23.9845182

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.94 + 172.68 = 174.61 seconds
