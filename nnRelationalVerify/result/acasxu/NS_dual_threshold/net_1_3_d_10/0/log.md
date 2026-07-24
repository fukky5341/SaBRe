## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 9159.82088535655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.52 + 2.46 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4030869, upper bound: 9164.4030869

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.4020495, upper bound: 9164.3983766
time: 0.75 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.49 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -9164.4020495, upper bound: 9164.3983766
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6288.5913086, 4675.7753906, -6180.7670898, 4591.1435547, -10879.7343750, 10856.5429688
1: -4999.2768555, 4495.6074219, -4913.7441406, 4414.3950195, -9413.6699219, 9409.3515625
2: -7260.2402344, 4891.1904297, -7136.4116211, 4799.6586914, -12059.8984375, 12027.6005859
3: -2738.5690918, 6978.8212891, -2687.8588867, 6858.0009766, -9596.5703125, 9666.6796875
4: -8066.1445312, 4850.3701172, -7928.8291016, 4759.1069336, -12825.2509766, 12779.1992188

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
time: 0.67 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
time: 0.72 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6396.1284180, 4760.1279297, -7047.5034180, 5201.9165039, -11598.0439453, 11807.6289062
1: -5084.7495117, 4576.8720703, -5601.0117188, 4998.3579102, -10083.1074219, 10177.8808594
2: -7382.6440430, 4981.4111328, -8125.0146484, 5435.6811523, -12818.3251953, 13106.4257812
3: -2788.3735352, 7098.8422852, -3041.9899902, 7791.6191406, -10579.9921875, 10140.8320312
4: -8202.3408203, 4940.7280273, -9027.1757812, 5397.2880859, -13599.6289062, 13967.9033203

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
time: 0.83 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.02 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -9164.3976090, upper bound: 9164.3976090

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -6180.7670898, 4591.1435547, -6180.7670898, 4591.1435547, -10771.9101562, 10771.9101562
1: -4913.7441406, 4414.3950195, -4913.7441406, 4414.3950195, -9328.1376953, 9328.1386719
2: -7136.4116211, 4799.6586914, -7136.4116211, 4799.6586914, -11936.0703125, 11936.0703125
3: -2687.8588867, 6858.0009766, -2687.8588867, 6858.0009766, -9545.8593750, 9545.8593750
4: -7928.8291016, 4759.1069336, -7928.8291016, 4759.1069336, -12687.9355469, 12687.9355469

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1195322, upper bound: 9163.8867644
time: 1.33 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.9711788
time: 0.81 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -6994.5791016, 5164.8876953, -6180.7670898, 4591.1435547, -11585.7216797, 11345.6533203
1: -5559.1464844, 4962.9487305, -4913.7441406, 4414.3950195, -9973.5410156, 9876.6933594
2: -8064.6806641, 5397.3959961, -7136.4116211, 4799.6586914, -12864.3398438, 12533.8066406
3: -3021.7844238, 7734.1054688, -2687.8588867, 6858.0009766, -9879.7851562, 10421.9609375
4: -8959.8271484, 5359.7524414, -7928.8291016, 4759.1069336, -13718.9335938, 13288.5800781

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1195322, upper bound: 9163.8867644
time: 0.70 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.9711788
time: 0.87 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6180.7670898, 4591.1435547, -7047.5034180, 5201.9165039, -11382.6826172, 11638.6464844
1: -4913.7441406, 4414.3950195, -5601.0117188, 4998.3579102, -9912.1015625, 10015.4042969
2: -7136.4116211, 4799.6586914, -8125.0146484, 5435.6811523, -12572.0927734, 12924.6738281
3: -2687.8588867, 6858.0009766, -3041.9899902, 7791.6191406, -10479.4765625, 9899.9912109
4: -7928.8291016, 4759.1069336, -9027.1757812, 5397.2880859, -13326.1171875, 13786.2822266

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8867644, upper bound: 9163.1170290
time: 0.71 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9711788, upper bound: 9163.9711783
time: 0.85 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -7047.5034180, 5201.9165039, -7047.5034180, 5201.9165039, -12249.4179688, 12249.4179688
1: -5601.0117188, 4998.3579102, -5601.0117188, 4998.3579102, -10599.3681641, 10599.3691406
2: -8125.0146484, 5435.6811523, -8125.0146484, 5435.6811523, -13560.6953125, 13560.6953125
3: -3041.9899902, 7791.6191406, -3041.9899902, 7791.6191406, -10833.6093750, 10833.6093750
4: -9027.1757812, 5397.2880859, -9027.1757812, 5397.2880859, -14424.4628906, 14424.4638672

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5830402, upper bound: 9160.0669808
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
time: 0.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.03 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.1195322, upper bound: 9163.8867644
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.9711788
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.1195322, upper bound: 9163.8867644
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.9711788
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.8867644, upper bound: 9163.1170290
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9163.9711788, upper bound: 9163.9711783
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9162.5830402, upper bound: 9160.0669808
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5881.3037109, 4383.3203125, -5964.6425781, 4439.7617188, -10321.0644531, 10347.9609375
1: -4675.1875000, 4213.2670898, -4741.9130859, 4268.9921875, -8944.1796875, 8955.1796875
2: -6794.1870117, 4586.6953125, -6887.9453125, 4643.0795898, -11437.2656250, 11474.6406250
3: -2574.3132324, 6533.6396484, -2603.2570801, 6623.7436523, -9198.0566406, 9136.8964844
4: -7550.4038086, 4551.4624023, -7653.7900391, 4604.0405273, -12154.4443359, 12205.2519531

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9074829, upper bound: 9161.8554945
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2883057, upper bound: 9161.8554945
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -6037.4443359, 4481.7182617, -6088.1240234, 4520.5644531, -10558.0068359, 10569.8417969
1: -4800.2705078, 4309.2353516, -4840.4609375, 4346.6816406, -9146.9521484, 9149.6962891
2: -6972.2402344, 4687.4731445, -7030.1733398, 4727.4482422, -11699.6884766, 11717.6464844
3: -2626.3544922, 6696.6015625, -2648.4008789, 6754.0366211, -9380.3896484, 9345.0019531
4: -7744.9453125, 4647.7915039, -7809.7163086, 4687.5097656, -12432.4550781, 12457.5078125

Time for backsubstitution: 0.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.8973804
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9742992, upper bound: 9163.9742997
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -6748.1435547, 4996.6962891, -5964.6425781, 4439.7617188, -11187.9023438, 10961.3388672
1: -5362.9433594, 4800.2612305, -4741.9130859, 4268.9921875, -9631.9326172, 9542.1738281
2: -7783.1577148, 5226.8666992, -6887.9453125, 4643.0795898, -12426.2363281, 12114.8125000
3: -2928.8874512, 7467.1240234, -2603.2570801, 6623.7436523, -9552.6308594, 10070.3798828
4: -8647.4335938, 5193.8886719, -7653.7900391, 4604.0405273, -13251.4736328, 12847.6787109

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5442596, upper bound: 9163.3544844
time: 0.81 seconds

## Relational analysis of NS_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1348319, upper bound: 9162.8573590
time: 0.79 seconds

## Relational analysis of NS_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7937492, upper bound: 9163.2823247
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6777.3178711, 5013.1181641, -6088.1240234, 4520.5644531, -11297.8779297, 11101.2421875
1: -5387.0805664, 4817.4008789, -4840.4609375, 4346.6816406, -9733.7617188, 9657.8613281
2: -7816.2412109, 5241.5922852, -7030.1733398, 4727.4482422, -12543.6894531, 12271.7656250
3: -2937.3088379, 7499.0219727, -2648.4008789, 6754.0366211, -9691.3447266, 10147.4208984
4: -8683.5185547, 5204.7070312, -7809.7163086, 4687.5097656, -13371.0283203, 13014.4238281

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.8953893
time: 0.61 seconds

## Relational analysis of NS_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2234412, upper bound: 9163.5993108
time: 0.72 seconds

## Relational analysis of NS_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6060390, upper bound: 9163.6013788
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5964.6425781, 4439.7617188, -6807.4023438, 5037.9760742, -11002.6162109, 11247.1621094
1: -4741.9130859, 4268.9921875, -5409.8369141, 4839.7177734, -9581.6298828, 9678.8281250
2: -6887.9453125, 4643.0795898, -7850.6411133, 5269.4223633, -12157.3671875, 12493.7207031
3: -2603.2570801, 6623.7436523, -2952.4069824, 7531.1806641, -10134.4375000, 9576.1503906
4: -7653.7900391, 4604.0405273, -8722.7529297, 5235.6293945, -12889.4179688, 13326.7919922

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3544844, upper bound: 9162.5442596
time: 0.66 seconds

## Relational analysis of NS_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8573590, upper bound: 9161.1348319
time: 0.65 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2823247, upper bound: 9161.7937492
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6088.1240234, 4520.5644531, -6830.5336914, 5050.7416992, -11138.8642578, 11351.0976562
1: -4840.4609375, 4346.6816406, -5429.1914062, 4853.4926758, -9693.9531250, 9775.8730469
2: -7030.1733398, 4727.4482422, -7876.9912109, 5280.9047852, -12311.0781250, 12604.4365234
3: -2648.4008789, 6754.0366211, -2958.0791016, 7557.2519531, -10205.6523438, 9712.1142578
4: -7809.7163086, 4687.5097656, -8751.5126953, 5243.0214844, -13052.7382812, 13439.0224609

Time for backsubstitution: 0.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8953893, upper bound: 9163.9748481
time: 0.73 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9710666, upper bound: 9163.9742992
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -6816.0756836, 5027.2036133, -6901.1176758, 5091.5351562, -11907.6103516, 11928.3203125
1: -5417.4545898, 4830.5981445, -5484.9033203, 4892.3129883, -10309.7675781, 10315.5009766
2: -7858.4067383, 5249.4301758, -7956.4682617, 5317.9428711, -13176.3496094, 13205.8974609
3: -2937.2355957, 7535.5585938, -2975.5261230, 7629.6464844, -10566.8818359, 10511.0839844
4: -8731.7285156, 5213.2846680, -8840.4414062, 5280.9580078, -14012.6845703, 14053.7236328

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -6641.0112305, 4908.4628906, -6876.6103516, 5073.4682617, -11714.4785156, 11785.0722656
1: -5278.4697266, 4717.0288086, -5465.5107422, 4874.4838867, -10152.9531250, 10182.5390625
2: -7655.4750977, 5136.7924805, -7929.2724609, 5297.1850586, -12952.6601562, 13066.0644531
3: -2874.3696289, 7331.1665039, -2966.0878906, 7603.3295898, -10477.6992188, 10297.2539062
4: -8501.7636719, 5092.6596680, -8809.7353516, 5259.9477539, -13761.7109375, 13902.3935547

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.57 seconds
NS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9162.9074829, upper bound: 9161.8554945
NS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9162.2883057, upper bound: 9161.8554945
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.9748481, upper bound: 9163.8973804
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.9742992, upper bound: 9163.9742997
NS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9161.1348319, upper bound: 9162.8573590
NS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9161.7937492, upper bound: 9163.2823247
NS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.2234412, upper bound: 9163.5993108
NS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.6060390, upper bound: 9163.6013788
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9162.8573590, upper bound: 9161.1348319
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.2823247, upper bound: 9161.7937492
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.8953893, upper bound: 9163.9748481
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9163.9710666, upper bound: 9163.9742992
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -9160.0560408, upper bound: 9160.0560408

## BFS NS instance: NS_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -5772.8886719, 4300.3681641, -5905.1171875, 4394.0126953, -10166.8984375, 10205.4853516
1: -4589.0747070, 4133.1611328, -4694.6489258, 4224.8505859, -8813.9257812, 8827.8085938
2: -6669.3852539, 4500.8295898, -6819.4121094, 4595.5976562, -11264.9804688, 11320.2392578
3: -2526.9184570, 6411.8354492, -2577.0112305, 6556.9648438, -9083.8828125, 8988.8466797
4: -7412.1601562, 4466.9008789, -7577.8515625, 4557.1870117, -11969.3437500, 12044.7500000

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2883057, upper bound: 9161.8554945
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2883057, upper bound: 9161.8554945
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -6017.9667969, 4498.3310547, -5864.7924805, 4364.7446289, -10382.7109375, 10363.1230469
1: -4785.3886719, 4327.6157227, -4662.5898438, 4196.9204102, -8982.3085938, 8990.2050781
2: -6954.8901367, 4713.0966797, -6772.6948242, 4563.3120117, -11518.2021484, 11485.7910156
3: -2639.5808105, 6696.3974609, -2558.9343262, 6513.2626953, -9152.8437500, 9255.3320312
4: -7728.1186523, 4667.7456055, -7526.0039062, 4525.0874023, -12253.2060547, 12193.7490234

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0359182, upper bound: 9161.8545658
time: 0.94 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0364623, upper bound: 9161.6401173
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5970.8471680, 4431.4228516, -5984.2055664, 4441.7426758, -10412.5898438, 10415.6289062
1: -4747.3022461, 4260.8491211, -4757.8115234, 4270.8999023, -9018.2021484, 9018.6601562
2: -6895.3129883, 4634.3203125, -6910.1000977, 4644.1718750, -11539.4843750, 11544.4199219
3: -2596.8671875, 6622.0000000, -2602.1740723, 6637.5239258, -9234.3906250, 9224.1738281
4: -7659.3857422, 4595.3232422, -7676.1884766, 4605.2578125, -12264.6425781, 12271.5087891

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6513241, upper bound: 9161.2106604
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4046731, upper bound: 9161.1376893
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5962.4213867, 4426.0747070, -6275.2075195, 4678.5048828, -10640.9248047, 10701.2822266
1: -4740.5854492, 4255.3842773, -4989.2094727, 4497.5913086, -9238.1767578, 9244.5937500
2: -6886.2836914, 4629.1250000, -7246.7749023, 4892.5678711, -11778.8515625, 11875.8984375
3: -2594.7824707, 6613.6215820, -2741.1262207, 6969.8789062, -9564.6611328, 9354.7480469
4: -7649.7348633, 4590.5170898, -8054.3598633, 4851.6313477, -12501.3662109, 12644.8769531

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1063251, upper bound: 9160.4033703
time: 0.66 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4019564, upper bound: 9160.4019569
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -5846.1054688, 4338.3515625, -5461.7988281, 4077.0751953, -9923.1796875, 9800.1494141
1: -4650.1030273, 4167.0273438, -4344.6313477, 3919.5600586, -8569.6611328, 8511.6572266
2: -6753.5566406, 4540.4228516, -6313.4257812, 4269.0908203, -11022.6474609, 10853.8466797
3: -2536.7287598, 6475.6611328, -2393.2470703, 6070.8798828, -8607.6083984, 8868.9082031
4: -7500.9033203, 4505.6894531, -7013.2011719, 4230.6059570, -11731.5078125, 11518.8906250

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_A1_A1_A1

### Relational analysis result of NS_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8050620, upper bound: 9162.5899358
time: 0.75 seconds

## Relational analysis of NS_B1_A2_A1_A1_A2

### Relational analysis result of NS_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.9301834, upper bound: 9162.6333119
time: 0.71 seconds

## BFS NS instance: NS_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -6659.8994141, 4930.4921875, -5911.6425781, 4400.1923828, -11060.0898438, 10842.1347656
1: -5292.7387695, 4736.7573242, -4699.9702148, 4230.9169922, -9523.6562500, 9436.7255859
2: -7681.0512695, 5158.3227539, -6826.5102539, 4601.8891602, -12282.9404297, 11984.8330078
3: -2891.7976074, 7368.6616211, -2580.6860352, 6564.4487305, -9456.2460938, 9949.3476562
4: -8534.1347656, 5126.3540039, -7585.5546875, 4563.5312500, -13097.6660156, 12711.9082031

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_A1_A2_A1

### Relational analysis result of NS_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3296275, upper bound: 9162.6786821
time: 0.85 seconds

## Relational analysis of NS_B1_A2_A1_A2_A2

### Relational analysis result of NS_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6528521, upper bound: 9163.0246637
time: 0.70 seconds

## BFS NS instance: NS_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -6499.3583984, 4815.5126953, -5887.2900391, 4382.7231445, -10882.0800781, 10702.8007812
1: -5166.2910156, 4627.7802734, -4681.0024414, 4213.5283203, -9379.8173828, 9308.7832031
2: -7496.9780273, 5037.7182617, -6799.7143555, 4584.2436523, -12081.2207031, 11837.4306641
3: -2827.1062012, 7192.7504883, -2569.1799316, 6535.0927734, -9362.1972656, 9761.9306641
4: -8329.7226562, 5002.3681641, -7555.0322266, 4544.6972656, -12874.4179688, 12557.4003906

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_A1_A1

### Relational analysis result of NS_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8519298, upper bound: 9162.7685906
time: 0.69 seconds

## Relational analysis of NS_B1_A2_A2_A1_A2

### Relational analysis result of NS_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1587106, upper bound: 9163.5257967
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -6522.4482422, 4828.5893555, -5962.9492188, 4424.2373047, -10946.6855469, 10791.5380859
1: -5184.5493164, 4639.8256836, -4740.9208984, 4253.4975586, -9438.0468750, 9380.7460938
2: -7525.2680664, 5051.5786133, -6887.0683594, 4627.4638672, -12152.7324219, 11938.6445312
3: -2831.7084961, 7218.1391602, -2592.9340820, 6611.8598633, -9443.5683594, 9811.0722656
4: -8360.5869141, 5014.8183594, -7650.5932617, 4588.2070312, -12948.7939453, 12665.4111328

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_A2_A1

### Relational analysis result of NS_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8924790, upper bound: 9163.4013832
time: 0.72 seconds

## Relational analysis of NS_B1_A2_A2_A2_A2

### Relational analysis result of NS_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5721662, upper bound: 9163.5700473
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5461.7988281, 4077.0751953, -5893.3984375, 4371.8095703, -9833.6083984, 9970.4716797
1: -4344.6313477, 3919.5600586, -4687.6499023, 4199.0688477, -8543.6992188, 8607.2080078
2: -6313.4257812, 4269.0908203, -6807.8378906, 4575.0087891, -10888.4326172, 11076.9287109
3: -2393.2470703, 6070.8798828, -2555.9370117, 6527.6645508, -8920.9121094, 8626.8164062
4: -7013.2011719, 4230.6059570, -7561.4643555, 4539.4702148, -11552.6718750, 11792.0664062

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5899358, upper bound: 9160.8050620
time: 0.67 seconds

## Relational analysis of NS_B2_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6333119, upper bound: 9160.9301834
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5911.6425781, 4400.1923828, -6701.8569336, 4959.6035156, -10871.2460938, 11102.0468750
1: -4699.9702148, 4230.9169922, -5325.9404297, 4764.5859375, -9464.5546875, 9556.8574219
2: -6826.5102539, 4601.8891602, -7728.8120117, 5188.3027344, -12014.8125000, 12330.7011719
3: -2580.6860352, 6564.4487305, -2908.2792969, 7413.8974609, -9994.5839844, 9472.7265625
4: -7585.5546875, 4563.5312500, -8587.4453125, 5155.7709961, -12741.3251953, 13150.9765625

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6786821, upper bound: 9160.3296275
time: 0.73 seconds

## Relational analysis of NS_B2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0246637, upper bound: 9161.6528521
time: 0.68 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5984.2055664, 4441.7426758, -6735.3208008, 4980.8754883, -10965.0810547, 11177.0634766
1: -4757.8115234, 4270.8999023, -5353.5761719, 4786.2128906, -9544.0244141, 9624.4746094
2: -6910.1000977, 4644.1718750, -7767.8330078, 5207.3642578, -12117.4628906, 12412.0039062
3: -2602.1740723, 6637.5239258, -2918.4167480, 7452.0390625, -10054.2119141, 9555.9404297
4: -7676.1884766, 4605.2578125, -8630.1650391, 5170.7041016, -12846.8906250, 13235.4228516

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4908454, upper bound: 9163.1259132
time: 0.68 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4950654, upper bound: 9163.6060390
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6275.2075195, 4678.5048828, -6733.0693359, 4978.4843750, -11253.6914062, 11411.5742188
1: -4989.2094727, 4497.5913086, -5352.2280273, 4784.0673828, -9773.2753906, 9849.8193359
2: -7246.7749023, 4892.5678711, -7765.1054688, 5205.0014648, -12451.7744141, 12657.6728516
3: -2741.1262207, 6969.8789062, -2916.3959961, 7449.4204102, -10190.5468750, 9886.2753906
4: -8054.3598633, 4851.6313477, -8626.8505859, 5168.0727539, -13222.4306641, 13478.4824219

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5815379, upper bound: 9163.1241805
time: 0.79 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6013788, upper bound: 9163.6059717
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6816.0756836, 5027.2036133, -6816.0756836, 5027.2036133, -11843.2783203, 11843.2783203
1: -5417.4545898, 4830.5981445, -5417.4545898, 4830.5981445, -10248.0527344, 10248.0527344
2: -7858.4067383, 5249.4301758, -7858.4067383, 5249.4301758, -13107.8369141, 13107.8369141
3: -2937.2355957, 7535.5585938, -2937.2355957, 7535.5585938, -10472.7939453, 10472.7939453
4: -8731.7285156, 5213.2846680, -8731.7285156, 5213.2846680, -13945.0136719, 13945.0136719

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6816.0756836, 5027.2036133, -6641.0112305, 4908.4628906, -11724.5380859, 11668.2138672
1: -5417.4545898, 4830.5981445, -5278.4697266, 4717.0288086, -10134.4833984, 10109.0683594
2: -7858.4067383, 5249.4301758, -7655.4750977, 5136.7924805, -12995.1992188, 12904.9052734
3: -2937.2355957, 7535.5585938, -2874.3696289, 7331.1665039, -10268.4003906, 10409.9277344
4: -8731.7285156, 5213.2846680, -8501.7636719, 5092.6596680, -13824.3886719, 13715.0488281

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5801590, upper bound: 9160.0668689
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6641.0112305, 4908.4628906, -6816.0756836, 5027.2036133, -11668.2138672, 11724.5371094
1: -5278.4697266, 4717.0288086, -5417.4545898, 4830.5981445, -10109.0683594, 10134.4833984
2: -7655.4750977, 5136.7924805, -7858.4067383, 5249.4301758, -12904.9042969, 12995.1992188
3: -2874.3696289, 7331.1665039, -2937.2355957, 7535.5585938, -10409.9277344, 10268.4013672
4: -8501.7636719, 5092.6596680, -8731.7285156, 5213.2846680, -13715.0488281, 13824.3886719

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6641.0112305, 4908.4628906, -6641.0112305, 4908.4628906, -11549.4736328, 11549.4726562
1: -5278.4697266, 4717.0288086, -5278.4697266, 4717.0288086, -9995.4980469, 9995.4980469
2: -7655.4750977, 5136.7924805, -7655.4750977, 5136.7924805, -12792.2675781, 12792.2675781
3: -2874.3696289, 7331.1665039, -2874.3696289, 7331.1665039, -10205.5341797, 10205.5341797
4: -8501.7636719, 5092.6596680, -8501.7636719, 5092.6596680, -13594.4238281, 13594.4238281

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.98 + 88.70 = 91.68 seconds
