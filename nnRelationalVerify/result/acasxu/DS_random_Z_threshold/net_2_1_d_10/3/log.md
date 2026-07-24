## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.8512133750000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292)
1: (-1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992)
2: (-1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595)
3: (-1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442)
4: (-1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.04 + 0.95 = 1.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.8699125, upper bound: 1.8699125

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8662006, upper bound: 1.8661717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8661717, upper bound: 1.8662006
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.52 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -1.8662006, upper bound: 1.8661717
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -1.8661717, upper bound: 1.8662006

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8599949, upper bound: 1.8652763
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8647611, upper bound: 1.8638599
time: 0.25 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8611965, upper bound: 1.8610762
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8611964, upper bound: 1.8661707
time: 0.24 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.77
Output dim: 0, lower bound: -1.8599949, upper bound: 1.8652763
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.77
Output dim: 0, lower bound: -1.8647611, upper bound: 1.8638599
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.77
Output dim: 0, lower bound: -1.8611965, upper bound: 1.8610762
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.77
Output dim: 0, lower bound: -1.8611964, upper bound: 1.8661707

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547935, upper bound: 1.8652432
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598909, upper bound: 1.8503017
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8512066, upper bound: 1.8637628
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8563042
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8576628, upper bound: 1.8610662
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8478651, upper bound: 1.8565294
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8611964, upper bound: 1.8648287
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598207, upper bound: 1.8661706
time: 0.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.46 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8547935, upper bound: 1.8652432
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8598909, upper bound: 1.8503017
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8512066, upper bound: 1.8637628
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8563042
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8576628, upper bound: 1.8610662
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8478651, upper bound: 1.8565294
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8611964, upper bound: 1.8648287
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.46
Output dim: 0, lower bound: -1.8598207, upper bound: 1.8661706

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547935, upper bound: 1.8652082
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547877, upper bound: 1.8652432
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8592278, upper bound: 1.8389458
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598797, upper bound: 1.8503017
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593785
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8502637, upper bound: 1.8637450
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8641436, upper bound: 1.8563042
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8550560
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8558961, upper bound: 1.8610662
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8576628, upper bound: 1.8609061
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8661204, upper bound: 1.8565235
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8660961, upper bound: 1.8565293
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8594870, upper bound: 1.8639322
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589069, upper bound: 1.8593389
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8571502, upper bound: 1.8661663
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8595468, upper bound: 1.8661706
time: 0.26 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8547935, upper bound: 1.8652082
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8547877, upper bound: 1.8652432
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8592278, upper bound: 1.8389458
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8598797, upper bound: 1.8503017
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593785
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8502637, upper bound: 1.8637450
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8641436, upper bound: 1.8563042
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8550560
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8558961, upper bound: 1.8610662
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8576628, upper bound: 1.8609061
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8661204, upper bound: 1.8565235
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8660961, upper bound: 1.8565293
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8594870, upper bound: 1.8639322
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8589069, upper bound: 1.8593389
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8571502, upper bound: 1.8661663
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -1.8595468, upper bound: 1.8661706

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8389170, upper bound: 1.8484542
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547213, upper bound: 1.8651820
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8588027
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547167, upper bound: 1.8652166
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8591551, upper bound: 1.8388669
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590823, upper bound: 1.8388669
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598797, upper bound: 1.8440543
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8593042, upper bound: 1.8503017
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8408931, upper bound: 1.8580335
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593785
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8511345, upper bound: 1.8637450
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8485435, upper bound: 1.8622613
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8641436, upper bound: 1.8562733
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598055, upper bound: 1.8563042
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8535244
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8639238, upper bound: 1.8550560
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8590639
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8482386, upper bound: 1.8592424
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8576329, upper bound: 1.8588707
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8531966, upper bound: 1.8609061
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598208, upper bound: 1.8485435
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8652166, upper bound: 1.8547166
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589512, upper bound: 1.8565294
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8660961, upper bound: 1.8563198
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8591102, upper bound: 1.8639322
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8594733, upper bound: 1.8598426
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589013, upper bound: 1.8593389
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8588426, upper bound: 1.8564803
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8555247, upper bound: 1.8647318
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8485247, upper bound: 1.8599170
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8579744, upper bound: 1.8661521
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8579744, upper bound: 1.8425476
time: 0.27 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8389170, upper bound: 1.8484542
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8547213, upper bound: 1.8651820
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8588027
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8547167, upper bound: 1.8652166
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8591551, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8590823, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8598797, upper bound: 1.8440543
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8593042, upper bound: 1.8503017
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8408931, upper bound: 1.8580335
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593785
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8511345, upper bound: 1.8637450
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8485435, upper bound: 1.8622613
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8641436, upper bound: 1.8562733
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8598055, upper bound: 1.8563042
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8647012, upper bound: 1.8535244
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8639238, upper bound: 1.8550560
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8590639
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8482386, upper bound: 1.8592424
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8576329, upper bound: 1.8588707
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8531966, upper bound: 1.8609061
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8598208, upper bound: 1.8485435
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8652166, upper bound: 1.8547166
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8589512, upper bound: 1.8565294
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8660961, upper bound: 1.8563198
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8591102, upper bound: 1.8639322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8594733, upper bound: 1.8598426
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8589013, upper bound: 1.8593389
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8588426, upper bound: 1.8564803
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8555247, upper bound: 1.8647318
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8485247, upper bound: 1.8599170
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8579744, upper bound: 1.8661521
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.53
Output dim: 0, lower bound: -1.8579744, upper bound: 1.8425476

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8544961, upper bound: 1.8651820
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547056, upper bound: 1.8579694
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8445277, upper bound: 1.8587319
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8587915
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8445277, upper bound: 1.8587319
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8547010, upper bound: 1.8639303
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8591551, upper bound: 1.8388669
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8563784, upper bound: 1.8388669
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590823, upper bound: 1.8388669
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8554361, upper bound: 1.8388669
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8440187
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8592424, upper bound: 1.8431028
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8592665, upper bound: 1.8502637
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577408, upper bound: 1.8482386
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8408931, upper bound: 1.8580335
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8554361
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593627
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8589997
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8511187, upper bound: 1.8637450
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8561729
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8485277, upper bound: 1.8622613
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8592980
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8641125, upper bound: 1.8562095
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590647, upper bound: 1.8560873
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8562427
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8571951, upper bound: 1.8560464
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8646724, upper bound: 1.8534609
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8590638, upper bound: 1.8516537
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8638971, upper bound: 1.8549832
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8575168, upper bound: 1.8543046
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8575168
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8516537, upper bound: 1.8590639
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8482386, upper bound: 1.8577408
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8431028, upper bound: 1.8592425
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8571951
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8563795
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8590648
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8590823
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8592980, upper bound: 1.8388669
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598090, upper bound: 1.8485277
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8639303, upper bound: 1.8547008
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8535164
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8561729, upper bound: 1.8388669
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8579694, upper bound: 1.8547056
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8637450, upper bound: 1.8511187
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8544961
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8638971
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8589997, upper bound: 1.8388669
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8562427, upper bound: 1.8597761
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8593627, upper bound: 1.8419542
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8502637, upper bound: 1.8592665
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8587915, upper bound: 1.8464156
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8563784
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8587319, upper bound: 1.8445277
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8534609, upper bound: 1.8646724
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8554361, upper bound: 1.8388669
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8440187, upper bound: 1.8598090
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8484542, upper bound: 1.8388669
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8562095, upper bound: 1.8641125
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8591551
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8580335, upper bound: 1.8408931
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8388669
time: 0.25 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8544961, upper bound: 1.8651820
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8547056, upper bound: 1.8579694
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8445277, upper bound: 1.8587319
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8587915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8445277, upper bound: 1.8587319
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8547010, upper bound: 1.8639303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8591551, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8563784, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8590823, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8554361, upper bound: 1.8388669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8440187
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8592424, upper bound: 1.8431028
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8592665, upper bound: 1.8502637
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8577408, upper bound: 1.8482386
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8408931, upper bound: 1.8580335
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8554361
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8593627
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8589997
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8511187, upper bound: 1.8637450
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8561729
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8485277, upper bound: 1.8622613
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8592980
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8641125, upper bound: 1.8562095
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8590647, upper bound: 1.8560873
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8419542, upper bound: 1.8562427
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8571951, upper bound: 1.8560464
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8646724, upper bound: 1.8534609
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8590638, upper bound: 1.8516537
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8638971, upper bound: 1.8549832
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8575168, upper bound: 1.8543046
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8575168
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8516537, upper bound: 1.8590639
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8482386, upper bound: 1.8577408
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8431028, upper bound: 1.8592425
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8571951
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8563795
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8464156, upper bound: 1.8590648
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8590823
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8592980, upper bound: 1.8388669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8598090, upper bound: 1.8485277
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8639303, upper bound: 1.8547008
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8535164
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8561729, upper bound: 1.8388669
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8579694, upper bound: 1.8547056
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8637450, upper bound: 1.8511187
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8544961
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8543046, upper bound: 1.8638971
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8589997, upper bound: 1.8388669
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8562427, upper bound: 1.8597761
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8593627, upper bound: 1.8419542
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8502637, upper bound: 1.8592665
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8587915, upper bound: 1.8464156
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8563784
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8587319, upper bound: 1.8445277
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8534609, upper bound: 1.8646724
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8554361, upper bound: 1.8388669
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8440187, upper bound: 1.8598090
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8484542, upper bound: 1.8388669
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8562095, upper bound: 1.8641125
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8591551
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8580335, upper bound: 1.8408931
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.70
Output dim: 0, lower bound: -1.8388669, upper bound: 1.8388669

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8517499, upper bound: 1.8621456
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8498574, upper bound: 1.8523078
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8522894, upper bound: 1.8490816
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8492465, upper bound: 1.8490816
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8369960, upper bound: 1.8570758
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8432362, upper bound: 1.8497633
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8426704, upper bound: 1.8402427
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8368187, upper bound: 1.8563490
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8505586, upper bound: 1.8623815
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8485361, upper bound: 1.8585756
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8324970, upper bound: 1.8390932
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8327451, upper bound: 1.8412948
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8398603, upper bound: 1.8212371
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8390867, upper bound: 1.8212371
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8533895, upper bound: 1.8359979
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8531174, upper bound: 1.8369258
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8382294, upper bound: 1.8324918
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8545912, upper bound: 1.8335670
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8536943, upper bound: 1.8364352
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8393166, upper bound: 1.8373209
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8568395, upper bound: 1.8365745
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8403884, upper bound: 1.8396195
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8551844, upper bound: 1.8484539
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8573697, upper bound: 1.8388223
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8551825, upper bound: 1.8378568
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8398951, upper bound: 1.8450515
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8377454, upper bound: 1.8405631
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8365464, upper bound: 1.8555070
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8373209, upper bound: 1.8392838
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8526926
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8384237, upper bound: 1.8580366
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8378614, upper bound: 1.8588008
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8214487, upper bound: 1.8391040
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8213861, upper bound: 1.8391601
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8345001, upper bound: 1.8397889
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8346094, upper bound: 1.8435257
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8371848, upper bound: 1.8542887
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8359197, upper bound: 1.8421999
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8443784, upper bound: 1.8599512
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8468938, upper bound: 1.8605329
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8335670, upper bound: 1.8546350
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8324918, upper bound: 1.8390596
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8381874, upper bound: 1.8523059
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8624475, upper bound: 1.8541846
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8345001, upper bound: 1.8351018
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8391124, upper bound: 1.8351018
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8370960, upper bound: 1.8533551
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8581988, upper bound: 1.8542408
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8379998, upper bound: 1.8542765
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8465776, upper bound: 1.8392777
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8407934, upper bound: 1.8344684
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8395874, upper bound: 1.8343934
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8566431, upper bound: 1.8388314
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8403043, upper bound: 1.8486423
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8427044, upper bound: 1.8354423
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8397488, upper bound: 1.8354423
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8523617, upper bound: 1.8524734
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8558855, upper bound: 1.8498499
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8498499, upper bound: 1.8558855
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8524734, upper bound: 1.8523617
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8472247, upper bound: 1.8569322
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8471951, upper bound: 1.8571404
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8368992, upper bound: 1.8560238
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8469664, upper bound: 1.8550858
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8385040, upper bound: 1.8577473
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8382709, upper bound: 1.8586805
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8212371, upper bound: 1.8388420
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8353800, upper bound: 1.8391413
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8335670, upper bound: 1.8517920
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8324918, upper bound: 1.8376672
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8394488, upper bound: 1.8574189
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8543513, upper bound: 1.8567808
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8371848, upper bound: 1.8574372
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8360169, upper bound: 1.8541712
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8458234, upper bound: 1.8359197
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8575605, upper bound: 1.8371848
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8569855, upper bound: 1.8368299
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8585959, upper bound: 1.8451866
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8621726, upper bound: 1.8531482
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8613959, upper bound: 1.8458420
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8621850, upper bound: 1.8516136
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8635516, upper bound: 1.8515887
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8534375, upper bound: 1.8364352
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8413951, upper bound: 1.8373209
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8490816, upper bound: 1.8492465
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8490816, upper bound: 1.8522894
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8614493, upper bound: 1.8495867
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8620801, upper bound: 1.8372959
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8626961, upper bound: 1.8387078
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8603335, upper bound: 1.8517405
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8354423, upper bound: 1.8397488
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8348128, upper bound: 1.8427044
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8472021, upper bound: 1.8359444
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8573555, upper bound: 1.8371848
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8368032, upper bound: 1.8554104
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8522326, upper bound: 1.8530620
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8575846, upper bound: 1.8361328
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8568180, upper bound: 1.8371681
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8472792, upper bound: 1.8406096
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8567152
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8391601, upper bound: 1.8303507
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8391040, upper bound: 1.8301557
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8373209, upper bound: 1.8396658
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8536582
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8581355, upper bound: 1.8383863
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8572992, upper bound: 1.8385461
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8400783, upper bound: 1.8597172
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518616, upper bound: 1.8629268
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8510123, upper bound: 1.8369239
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8538807, upper bound: 1.8368550
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8405452, upper bound: 1.8406429
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8574190
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8535788, upper bound: 1.8511999
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8387661, upper bound: 1.8617337
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 31

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8369258, upper bound: 1.8570274
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8359979, upper bound: 1.8572254
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 0

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8501948, upper bound: 1.8361135
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8558339, upper bound: 1.8370605
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8517499, upper bound: 1.8621456
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8498574, upper bound: 1.8523078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8522894, upper bound: 1.8490816
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8492465, upper bound: 1.8490816
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8369960, upper bound: 1.8570758
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8432362, upper bound: 1.8497633
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8426704, upper bound: 1.8402427
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8368187, upper bound: 1.8563490
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8505586, upper bound: 1.8623815
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8485361, upper bound: 1.8585756
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8324970, upper bound: 1.8390932
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8327451, upper bound: 1.8412948
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8398603, upper bound: 1.8212371
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8390867, upper bound: 1.8212371
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8533895, upper bound: 1.8359979
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8531174, upper bound: 1.8369258
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8382294, upper bound: 1.8324918
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8545912, upper bound: 1.8335670
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8536943, upper bound: 1.8364352
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8393166, upper bound: 1.8373209
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8568395, upper bound: 1.8365745
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8403884, upper bound: 1.8396195
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8551844, upper bound: 1.8484539
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8573697, upper bound: 1.8388223
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8551825, upper bound: 1.8378568
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8398951, upper bound: 1.8450515
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8377454, upper bound: 1.8405631
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8365464, upper bound: 1.8555070
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8373209, upper bound: 1.8392838
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8526926
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8384237, upper bound: 1.8580366
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8378614, upper bound: 1.8588008
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8214487, upper bound: 1.8391040
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8213861, upper bound: 1.8391601
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8345001, upper bound: 1.8397889
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8346094, upper bound: 1.8435257
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8371848, upper bound: 1.8542887
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8359197, upper bound: 1.8421999
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8443784, upper bound: 1.8599512
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8468938, upper bound: 1.8605329
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8335670, upper bound: 1.8546350
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8324918, upper bound: 1.8390596
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8381874, upper bound: 1.8523059
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8624475, upper bound: 1.8541846
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8345001, upper bound: 1.8351018
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8391124, upper bound: 1.8351018
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8370960, upper bound: 1.8533551
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8581988, upper bound: 1.8542408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8379998, upper bound: 1.8542765
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8465776, upper bound: 1.8392777
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8407934, upper bound: 1.8344684
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8395874, upper bound: 1.8343934
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8566431, upper bound: 1.8388314
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8403043, upper bound: 1.8486423
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8427044, upper bound: 1.8354423
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8397488, upper bound: 1.8354423
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8523617, upper bound: 1.8524734
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8558855, upper bound: 1.8498499
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8498499, upper bound: 1.8558855
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8524734, upper bound: 1.8523617
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8472247, upper bound: 1.8569322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8471951, upper bound: 1.8571404
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8368992, upper bound: 1.8560238
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8469664, upper bound: 1.8550858
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8385040, upper bound: 1.8577473
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8382709, upper bound: 1.8586805
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8212371, upper bound: 1.8388420
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8353800, upper bound: 1.8391413
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8335670, upper bound: 1.8517920
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8324918, upper bound: 1.8376672
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8394488, upper bound: 1.8574189
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8543513, upper bound: 1.8567808
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8371848, upper bound: 1.8574372
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8360169, upper bound: 1.8541712
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8458234, upper bound: 1.8359197
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8575605, upper bound: 1.8371848
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8569855, upper bound: 1.8368299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8585959, upper bound: 1.8451866
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8621726, upper bound: 1.8531482
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8613959, upper bound: 1.8458420
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8621850, upper bound: 1.8516136
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8635516, upper bound: 1.8515887
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8534375, upper bound: 1.8364352
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8413951, upper bound: 1.8373209
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8490816, upper bound: 1.8492465
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8490816, upper bound: 1.8522894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8614493, upper bound: 1.8495867
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8620801, upper bound: 1.8372959
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8626961, upper bound: 1.8387078
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8603335, upper bound: 1.8517405
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8354423, upper bound: 1.8397488
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8348128, upper bound: 1.8427044
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8472021, upper bound: 1.8359444
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8573555, upper bound: 1.8371848
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8368032, upper bound: 1.8554104
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8522326, upper bound: 1.8530620
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8575846, upper bound: 1.8361328
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8568180, upper bound: 1.8371681
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8472792, upper bound: 1.8406096
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8567152
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8391601, upper bound: 1.8303507
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8391040, upper bound: 1.8301557
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8373209, upper bound: 1.8396658
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8536582
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8581355, upper bound: 1.8383863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8572992, upper bound: 1.8385461
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8400783, upper bound: 1.8597172
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8518616, upper bound: 1.8629268
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8510123, upper bound: 1.8369239
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8538807, upper bound: 1.8368550
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8405452, upper bound: 1.8406429
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8364352, upper bound: 1.8574190
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8535788, upper bound: 1.8511999
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8387661, upper bound: 1.8617337
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8369258, upper bound: 1.8570274
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8359979, upper bound: 1.8572254
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8501948, upper bound: 1.8361135
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.24
Output dim: 0, lower bound: -1.8558339, upper bound: 1.8370605

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8490000, upper bound: 1.8564895
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8349838, upper bound: 1.8595435
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8478997, upper bound: 1.8508559
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8472916, upper bound: 1.8403579
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8399848, upper bound: 1.8475164
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8507677, upper bound: 1.8431200
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8353297, upper bound: 1.8554226
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8350085, upper bound: 1.8516700
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8201817, upper bound: 1.8367347
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8201817, upper bound: 1.8367844
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8347184, upper bound: 1.8581443
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8461233, upper bound: 1.8535209
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8314672, upper bound: 1.8539091
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8444537, upper bound: 1.8411116
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8427841, upper bound: 1.8343930
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8514854, upper bound: 1.8343930
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8316057, upper bound: 1.8313370
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8482308, upper bound: 1.8318212
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8524814, upper bound: 1.8313370
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8524131, upper bound: 1.8318212
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8350835, upper bound: 1.8308142
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8475667, upper bound: 1.8308142
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8369790, upper bound: 1.8212859
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8367938, upper bound: 1.8214394
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8364214, upper bound: 1.8431407
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518606, upper bound: 1.8359593
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8539077, upper bound: 1.8343930
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8549069, upper bound: 1.8366691
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8523763, upper bound: 1.8363489
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8534114, upper bound: 1.8346462
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8546137
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8358119, upper bound: 1.8547280
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8344677, upper bound: 1.8511234
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8344677, upper bound: 1.8482333
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8334452, upper bound: 1.8539760
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8334462, upper bound: 1.8320165
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8364083, upper bound: 1.8357002
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8358270, upper bound: 1.8564241
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8197643, upper bound: 1.8370301
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8197643, upper bound: 1.8387314
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8394274, upper bound: 1.8570435
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8343930, upper bound: 1.8565972
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8435526, upper bound: 1.8569132
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8350601, upper bound: 1.8575684
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8329010, upper bound: 1.8541198
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8318258, upper bound: 1.8541972
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8368664, upper bound: 1.8372671
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8368608, upper bound: 1.8494616
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8368664, upper bound: 1.8385283
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8368608, upper bound: 1.8515281
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8360536, upper bound: 1.8380034
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8360495, upper bound: 1.8506317
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8552554, upper bound: 1.8389858
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8345962, upper bound: 1.8515611
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8394274, upper bound: 1.8516623
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518593, upper bound: 1.8480015
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8560817, upper bound: 1.8367864
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8554594, upper bound: 1.8357001
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8348947, upper bound: 1.8477827
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8473815, upper bound: 1.8353106
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8366397, upper bound: 1.8449816
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8512370, upper bound: 1.8342918
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8291924, upper bound: 1.8370413
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8296240, upper bound: 1.8370664
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8334323, upper bound: 1.8355829
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8333778, upper bound: 1.8358191
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8366756, upper bound: 1.8552033
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8454078, upper bound: 1.8538828
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8367516, upper bound: 1.8349770
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8367516, upper bound: 1.8546978
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8316951, upper bound: 1.8514905
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8311521, upper bound: 1.8359435
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8385301, upper bound: 1.8533217
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8385301, upper bound: 1.8535793
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8363367, upper bound: 1.8564236
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8353081, upper bound: 1.8376251
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8348869, upper bound: 1.8570183
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8366821, upper bound: 1.8536864
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8308142, upper bound: 1.8336331
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8308142, upper bound: 1.8489246
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8345512, upper bound: 1.8527859
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8348810, upper bound: 1.8455007
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8458235, upper bound: 1.8536767
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8516343, upper bound: 1.8545265
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8351836, upper bound: 1.8557771
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8352518, upper bound: 1.8515142
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8345208, upper bound: 1.8381475
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8343940, upper bound: 1.8515612
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8550867, upper bound: 1.8346671
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8484789, upper bound: 1.8356182
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8560092, upper bound: 1.8351197
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8564741, upper bound: 1.8350132
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8362296, upper bound: 1.8295259
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8351589, upper bound: 1.8291439
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8577603, upper bound: 1.8472368
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8594224, upper bound: 1.8502896
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8397931, upper bound: 1.8249331
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8376009, upper bound: 1.8243116
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8272657, upper bound: 1.8319475
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8372701, upper bound: 1.8318015
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8598555, upper bound: 1.8364934
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8585258, upper bound: 1.8487052
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8402475, upper bound: 1.8346671
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8396453, upper bound: 1.8346671
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8359293, upper bound: 1.8497703
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8475130, upper bound: 1.8503357
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8596111, upper bound: 1.8480381
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8592508, upper bound: 1.8416421
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8411696, upper bound: 1.8221570
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8359792, upper bound: 1.8206253
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8409950, upper bound: 1.8199290
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8199290, upper bound: 1.8199290
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8481491, upper bound: 1.8475090
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8553557, upper bound: 1.8344575
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8376706, upper bound: 1.8199175
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8376159, upper bound: 1.8199764
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8178282, upper bound: 1.8359824
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8178282, upper bound: 1.8370041
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8498545, upper bound: 1.8495420
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8495830, upper bound: 1.8476164
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8570318, upper bound: 1.8353010
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8376251, upper bound: 1.8352972
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8552071, upper bound: 1.8350140
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8392782, upper bound: 1.8354679
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8555314
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8361388, upper bound: 1.8555314
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8508077
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8508077
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8457816, upper bound: 1.8366290
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8564794, upper bound: 1.8353049
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8318258, upper bound: 1.8341846
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8531207, upper bound: 1.8338856
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8198468, upper bound: 1.8380306
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8198641, upper bound: 1.8392543
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8349802, upper bound: 1.8588985
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8472218, upper bound: 1.8543414
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8498601, upper bound: 1.8345598
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8498601, upper bound: 1.8360392
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8346671, upper bound: 1.8557602
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8348283, upper bound: 1.8546512
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8391106, upper bound: 1.8430780
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8520371, upper bound: 1.8495434
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8349770, upper bound: 1.8594642
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8371406, upper bound: 1.8561615
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8318212, upper bound: 1.8524940
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8313370, upper bound: 1.8315957
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8351926, upper bound: 1.8538960
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8351926, upper bound: 1.8566526
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8341140, upper bound: 1.8198082
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8341140, upper bound: 1.8206746
time: 0.29 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8490000, upper bound: 1.8564895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8349838, upper bound: 1.8595435
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8478997, upper bound: 1.8508559
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8472916, upper bound: 1.8403579
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8399848, upper bound: 1.8475164
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8507677, upper bound: 1.8431200
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8353297, upper bound: 1.8554226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8350085, upper bound: 1.8516700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8201817, upper bound: 1.8367347
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8201817, upper bound: 1.8367844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8347184, upper bound: 1.8581443
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8461233, upper bound: 1.8535209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8314672, upper bound: 1.8539091
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8444537, upper bound: 1.8411116
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8427841, upper bound: 1.8343930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8514854, upper bound: 1.8343930
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8316057, upper bound: 1.8313370
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8482308, upper bound: 1.8318212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8524814, upper bound: 1.8313370
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8524131, upper bound: 1.8318212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8350835, upper bound: 1.8308142
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8475667, upper bound: 1.8308142
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8369790, upper bound: 1.8212859
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8367938, upper bound: 1.8214394
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8364214, upper bound: 1.8431407
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8518606, upper bound: 1.8359593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8539077, upper bound: 1.8343930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8549069, upper bound: 1.8366691
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8523763, upper bound: 1.8363489
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8534114, upper bound: 1.8346462
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8546137
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8358119, upper bound: 1.8547280
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8344677, upper bound: 1.8511234
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8344677, upper bound: 1.8482333
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8334452, upper bound: 1.8539760
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8334462, upper bound: 1.8320165
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8364083, upper bound: 1.8357002
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8358270, upper bound: 1.8564241
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8197643, upper bound: 1.8370301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8197643, upper bound: 1.8387314
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8394274, upper bound: 1.8570435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8343930, upper bound: 1.8565972
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8435526, upper bound: 1.8569132
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8350601, upper bound: 1.8575684
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8329010, upper bound: 1.8541198
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8318258, upper bound: 1.8541972
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8368664, upper bound: 1.8372671
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8368608, upper bound: 1.8494616
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8368664, upper bound: 1.8385283
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8368608, upper bound: 1.8515281
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8360536, upper bound: 1.8380034
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8360495, upper bound: 1.8506317
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8552554, upper bound: 1.8389858
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8345962, upper bound: 1.8515611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8394274, upper bound: 1.8516623
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8518593, upper bound: 1.8480015
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8560817, upper bound: 1.8367864
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8554594, upper bound: 1.8357001
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8348947, upper bound: 1.8477827
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8473815, upper bound: 1.8353106
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8366397, upper bound: 1.8449816
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8512370, upper bound: 1.8342918
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8291924, upper bound: 1.8370413
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8296240, upper bound: 1.8370664
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8334323, upper bound: 1.8355829
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8333778, upper bound: 1.8358191
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8366756, upper bound: 1.8552033
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8454078, upper bound: 1.8538828
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8367516, upper bound: 1.8349770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8367516, upper bound: 1.8546978
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8316951, upper bound: 1.8514905
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8311521, upper bound: 1.8359435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8385301, upper bound: 1.8533217
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8385301, upper bound: 1.8535793
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8363367, upper bound: 1.8564236
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8353081, upper bound: 1.8376251
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8348869, upper bound: 1.8570183
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8366821, upper bound: 1.8536864
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8308142, upper bound: 1.8336331
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8308142, upper bound: 1.8489246
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8345512, upper bound: 1.8527859
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8348810, upper bound: 1.8455007
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8458235, upper bound: 1.8536767
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8516343, upper bound: 1.8545265
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8351836, upper bound: 1.8557771
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8352518, upper bound: 1.8515142
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8345208, upper bound: 1.8381475
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8343940, upper bound: 1.8515612
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8550867, upper bound: 1.8346671
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8484789, upper bound: 1.8356182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8560092, upper bound: 1.8351197
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8564741, upper bound: 1.8350132
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8362296, upper bound: 1.8295259
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8351589, upper bound: 1.8291439
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8577603, upper bound: 1.8472368
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8594224, upper bound: 1.8502896
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8397931, upper bound: 1.8249331
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8376009, upper bound: 1.8243116
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8272657, upper bound: 1.8319475
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8372701, upper bound: 1.8318015
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8598555, upper bound: 1.8364934
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8585258, upper bound: 1.8487052
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8402475, upper bound: 1.8346671
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8396453, upper bound: 1.8346671
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8359293, upper bound: 1.8497703
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8475130, upper bound: 1.8503357
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8596111, upper bound: 1.8480381
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8592508, upper bound: 1.8416421
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8411696, upper bound: 1.8221570
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8359792, upper bound: 1.8206253
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8409950, upper bound: 1.8199290
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8199290, upper bound: 1.8199290
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8481491, upper bound: 1.8475090
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8553557, upper bound: 1.8344575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8376706, upper bound: 1.8199175
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8376159, upper bound: 1.8199764
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8178282, upper bound: 1.8359824
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8178282, upper bound: 1.8370041
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8498545, upper bound: 1.8495420
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8495830, upper bound: 1.8476164
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8570318, upper bound: 1.8353010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8376251, upper bound: 1.8352972
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8552071, upper bound: 1.8350140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8392782, upper bound: 1.8354679
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8555314
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8361388, upper bound: 1.8555314
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8508077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8357001, upper bound: 1.8508077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8457816, upper bound: 1.8366290
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8564794, upper bound: 1.8353049
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8318258, upper bound: 1.8341846
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8531207, upper bound: 1.8338856
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8198468, upper bound: 1.8380306
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8198641, upper bound: 1.8392543
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8349802, upper bound: 1.8588985
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8472218, upper bound: 1.8543414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8498601, upper bound: 1.8345598
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8498601, upper bound: 1.8360392
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8346671, upper bound: 1.8557602
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8348283, upper bound: 1.8546512
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8391106, upper bound: 1.8430780
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8520371, upper bound: 1.8495434
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8349770, upper bound: 1.8594642
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8371406, upper bound: 1.8561615
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8318212, upper bound: 1.8524940
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8313370, upper bound: 1.8315957
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8351926, upper bound: 1.8538960
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8351926, upper bound: 1.8566526
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8341140, upper bound: 1.8198082
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.01
Output dim: 0, lower bound: -1.8341140, upper bound: 1.8206746

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8332245, upper bound: 1.8545678
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8473914, upper bound: 1.8547495
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8332245, upper bound: 1.8564058
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8332311, upper bound: 1.8577567
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8301884, upper bound: 1.8507704
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8290414, upper bound: 1.8312320
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8341667, upper bound: 1.8458499
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8336646, upper bound: 1.8466569
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8148597, upper bound: 1.8330620
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8148597, upper bound: 1.8357117
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8422865, upper bound: 1.8435144
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8392594, upper bound: 1.8409408
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8296192, upper bound: 1.8295285
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8295285, upper bound: 1.8515578
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8472239, upper bound: 1.8335934
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8445240, upper bound: 1.8335934
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8468310, upper bound: 1.8297959
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8506991, upper bound: 1.8297959
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8501213, upper bound: 1.8295285
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8336419, upper bound: 1.8295285
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8491835, upper bound: 1.8292241
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8331694, upper bound: 1.8326788
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8339720, upper bound: 1.8297959
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8490429, upper bound: 1.8297959
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8319548, upper bound: 1.8307547
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8500907, upper bound: 1.8318082
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8324909, upper bound: 1.8312166
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8458514, upper bound: 1.8291589
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8522574, upper bound: 1.8338941
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8518163, upper bound: 1.8337456
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8342878, upper bound: 1.8526899
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8342878, upper bound: 1.8456014
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8300034, upper bound: 1.8505247
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8300547, upper bound: 1.8340330
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8300034, upper bound: 1.8340256
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8300034, upper bound: 1.8517477
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8343287, upper bound: 1.8546548
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8343983, upper bound: 1.8546551
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8333664, upper bound: 1.8553786
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8376865, upper bound: 1.8548933
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8335934, upper bound: 1.8546428
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8335934, upper bound: 1.8561263
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Candidate
type: DSZ, layer: 5, pos: 35

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8334441, upper bound: 1.8528329
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8419085, upper bound: 1.8551860
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8329189, upper bound: 1.8555722
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8331216, upper bound: 1.8559035
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8311419, upper bound: 1.8522104
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8306577, upper bound: 1.8519869
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8306577, upper bound: 1.8522364
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.8306577, upper bound: 1.8522674
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8437771, upper bound: 1.8475463
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8454099, upper bound: 1.8322073
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8491757, upper bound: 1.8361463
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8491757, upper bound: 1.8337456
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 33
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8197071, upper bound: 1.8267006
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8197071, upper bound: 1.8259152
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8471706, upper bound: 1.8441331
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8343205, upper bound: 1.8437153
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 16
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 35
type: DSZ, layer: 5, pos: 22
type: DSZ, layer: 5, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8374832, upper bound: 1.8462239
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8501940, upper bound: 1.8442938
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.5446405, 0.9670887, -1.5446405, 0.9670887, -2.5117292, 2.5117292
1: -1.4420700, 0.8331291, -1.4420700, 0.8331291, -2.2751992, 2.2751992
2: -1.1730814, 0.8038781, -1.1730814, 0.8038781, -1.9769595, 1.9769595
3: -1.4580498, 0.8813943, -1.4580498, 0.8813943, -2.3394442, 2.3394442
4: -1.3076820, 0.9560763, -1.3076820, 0.9560763, -2.2637582, 2.2637582

Time for backsubstitution: 1.41 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 1.99 + 418.40 = 420.39 seconds
