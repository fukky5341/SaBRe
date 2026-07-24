## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 268.920435005728


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421)
1: (-69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938)
2: (-76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445)
3: (-68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744)
4: (-111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 1.76 = 2.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -268.9365712, upper bound: 268.9365712

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9318499, upper bound: 268.9259371
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9259371, upper bound: 268.9318499
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.25 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -268.9318499, upper bound: 268.9259371
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -268.9259371, upper bound: 268.9318499

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9177338, upper bound: 268.9209896
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9251744, upper bound: 268.9177338
time: 0.61 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9258766, upper bound: 268.9252569
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9208451, upper bound: 268.9318499
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 4, lower bound: -268.9177338, upper bound: 268.9209896
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 4, lower bound: -268.9251744, upper bound: 268.9177338
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 4, lower bound: -268.9258766, upper bound: 268.9252569
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 4, lower bound: -268.9208451, upper bound: 268.9318499

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9251023, upper bound: 268.9172797
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9172797, upper bound: 268.9208425
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9187446, upper bound: 268.9104495
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9104784, upper bound: 268.9109888
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9202864, upper bound: 268.9202736
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9202736, upper bound: 268.9244534
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9205103, upper bound: 268.9282100
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9205103, upper bound: 268.9317495
time: 1.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9251023, upper bound: 268.9172797
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9172797, upper bound: 268.9208425
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9187446, upper bound: 268.9104495
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9104784, upper bound: 268.9109888
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9202864, upper bound: 268.9202736
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9202736, upper bound: 268.9244534
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9205103, upper bound: 268.9282100
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.76
Output dim: 4, lower bound: -268.9205103, upper bound: 268.9317495

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9172318, upper bound: 268.9172318
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9249760, upper bound: 268.9172318
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9138677, upper bound: 268.9188529
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9138267, upper bound: 268.9187553
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9199539, upper bound: 268.9199539
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9199539, upper bound: 268.9242599
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9188829, upper bound: 268.9189066
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9188829, upper bound: 268.9264031
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9127440, upper bound: 268.9191459
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9127440, upper bound: 268.9217301
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9172318, upper bound: 268.9172318
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9249760, upper bound: 268.9172318
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9138677, upper bound: 268.9188529
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9138267, upper bound: 268.9187553
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9199539, upper bound: 268.9199539
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9199539, upper bound: 268.9242599
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9188829, upper bound: 268.9189066
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9188829, upper bound: 268.9264031
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9127440, upper bound: 268.9191459
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 4, lower bound: -268.9127440, upper bound: 268.9217301

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9237284, upper bound: 268.9133965
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9134431, upper bound: 268.9133965
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9190627, upper bound: 268.9195370
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9190627, upper bound: 268.9238520
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9180216, upper bound: 268.9180216
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9180216, upper bound: 268.9261567
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9116826, upper bound: 268.9116826
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9116826, upper bound: 268.9216767
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9237284, upper bound: 268.9133965
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9134431, upper bound: 268.9133965
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9190627, upper bound: 268.9195370
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9190627, upper bound: 268.9238520
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9180216, upper bound: 268.9180216
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9180216, upper bound: 268.9261567
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9116826, upper bound: 268.9116826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 4, lower bound: -268.9116826, upper bound: 268.9216767

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9140754, upper bound: 268.9133685
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9140754, upper bound: 268.9133685
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9108477, upper bound: 268.9185824
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9108477, upper bound: 268.9185824
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9146465, upper bound: 268.9146465
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9146465, upper bound: 268.9250490
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9108806, upper bound: 268.9172964
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9108806, upper bound: 268.9183483
time: 0.84 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9140754, upper bound: 268.9133685
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9140754, upper bound: 268.9133685
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9108477, upper bound: 268.9185824
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9108477, upper bound: 268.9185824
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9146465, upper bound: 268.9146465
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9146465, upper bound: 268.9250490
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9108806, upper bound: 268.9172964
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.35
Output dim: 4, lower bound: -268.9108806, upper bound: 268.9183483

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9145947, upper bound: 268.9247553
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9145947, upper bound: 268.9246313
time: 0.60 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.82 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 4, lower bound: -268.9145947, upper bound: 268.9247553
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 4, lower bound: -268.9145947, upper bound: 268.9246313

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9110515, upper bound: 268.9110515
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9110515, upper bound: 268.9225019
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9142728, upper bound: 268.9142728
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9142728, upper bound: 268.9245100
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.81 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.81
Output dim: 4, lower bound: -268.9110515, upper bound: 268.9110515
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.81
Output dim: 4, lower bound: -268.9110515, upper bound: 268.9225019
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.81
Output dim: 4, lower bound: -268.9142728, upper bound: 268.9142728
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.81
Output dim: 4, lower bound: -268.9142728, upper bound: 268.9245100

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9108255, upper bound: 268.9114032
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9108255, upper bound: 268.9224452
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9141307, upper bound: 268.9141307
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9141307, upper bound: 268.9244407
time: 0.62 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.84 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.84
Output dim: 4, lower bound: -268.9108255, upper bound: 268.9114032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.84
Output dim: 4, lower bound: -268.9108255, upper bound: 268.9224452
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.84
Output dim: 4, lower bound: -268.9141307, upper bound: 268.9141307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.84
Output dim: 4, lower bound: -268.9141307, upper bound: 268.9244407

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9101805, upper bound: 268.9124706
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9101805, upper bound: 268.9216139
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9107354, upper bound: 268.9107354
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9107354, upper bound: 268.9223080
time: 0.57 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.77 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.77
Output dim: 4, lower bound: -268.9101805, upper bound: 268.9124706
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.77
Output dim: 4, lower bound: -268.9101805, upper bound: 268.9216139
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.77
Output dim: 4, lower bound: -268.9107354, upper bound: 268.9107354
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.77
Output dim: 4, lower bound: -268.9107354, upper bound: 268.9223080

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9062439, upper bound: 268.9186069
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9062439, upper bound: 268.9178517
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9100775, upper bound: 268.9101091
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9100775, upper bound: 268.9214898
time: 0.57 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 1.77 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.77
Output dim: 4, lower bound: -268.9062439, upper bound: 268.9186069
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.77
Output dim: 4, lower bound: -268.9062439, upper bound: 268.9178517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.77
Output dim: 4, lower bound: -268.9100775, upper bound: 268.9101091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.77
Output dim: 4, lower bound: -268.9100775, upper bound: 268.9214898

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421
1: -69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938
2: -76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445
3: -68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744
4: -111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9045154, upper bound: 268.9045154
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9045154, upper bound: 268.9095631
time: 0.52 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 2.05 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 2.05
Output dim: 4, lower bound: -268.9045154, upper bound: 268.9045154
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 2.05
Output dim: 4, lower bound: -268.9045154, upper bound: 268.9095631

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.39 + 61.84 = 64.23 seconds
