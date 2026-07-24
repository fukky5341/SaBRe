## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 155.85206263506


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228)
1: (-62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323)
2: (-55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172)
3: (-72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134)
4: (-72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.10 + 2.27 = 3.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -155.8676494, upper bound: 155.8676494

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8673899, upper bound: 155.8673899
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8673899, upper bound: 155.8673899
time: 1.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.02 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 4, lower bound: -155.8673899, upper bound: 155.8673899
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 4, lower bound: -155.8673899, upper bound: 155.8673899

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
time: 0.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 4, lower bound: -155.8672620, upper bound: 155.8672620

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8671049, upper bound: 155.8671030
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8671093, upper bound: 155.8671046
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569259, upper bound: 155.8569978
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569259, upper bound: 155.8569978
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672153, upper bound: 155.8672062
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8672153, upper bound: 155.8672448
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670664, upper bound: 155.8670870
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670870, upper bound: 155.8670664
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.68 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8671049, upper bound: 155.8671030
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8671093, upper bound: 155.8671046
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8569259, upper bound: 155.8569978
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8569259, upper bound: 155.8569978
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8672153, upper bound: 155.8672062
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8672153, upper bound: 155.8672448
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8670664, upper bound: 155.8670870
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 4, lower bound: -155.8670870, upper bound: 155.8670664

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670505, upper bound: 155.8670601
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670505, upper bound: 155.8670924
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8651973, upper bound: 155.8651973
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8651973, upper bound: 155.8652814
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8469965, upper bound: 155.8469965
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8469965, upper bound: 155.8469966
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569149, upper bound: 155.8569149
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569205, upper bound: 155.8569867
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670646, upper bound: 155.8670515
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8670725, upper bound: 155.8670515
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8662737, upper bound: 155.8662453
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8662448, upper bound: 155.8663619
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658483
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658788
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8669210, upper bound: 155.8668975
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8669194, upper bound: 155.8668974
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8670505, upper bound: 155.8670601
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8670505, upper bound: 155.8670924
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8651973, upper bound: 155.8651973
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8651973, upper bound: 155.8652814
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8469965, upper bound: 155.8469965
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8469965, upper bound: 155.8469966
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8569149, upper bound: 155.8569149
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8569205, upper bound: 155.8569867
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8670646, upper bound: 155.8670515
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8670725, upper bound: 155.8670515
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8662737, upper bound: 155.8662453
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8662448, upper bound: 155.8663619
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658483
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658788
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8669210, upper bound: 155.8668975
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 4, lower bound: -155.8669194, upper bound: 155.8668974

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8668787, upper bound: 155.8669037
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8668787, upper bound: 155.8668956
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8507324, upper bound: 155.8507752
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8507324, upper bound: 155.8507752
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564580, upper bound: 155.8565242
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564583, upper bound: 155.8565242
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468966, upper bound: 155.8468966
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468966, upper bound: 155.8468966
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569149, upper bound: 155.8569867
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8569205, upper bound: 155.8569149
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8610130, upper bound: 155.8610130
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8610274, upper bound: 155.8610130
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658118, upper bound: 155.8658041
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658053, upper bound: 155.8658041
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8646919, upper bound: 155.8646922
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8646919, upper bound: 155.8646925
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8662415, upper bound: 155.8662363
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8662399, upper bound: 155.8663496
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656449
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656073
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8589362, upper bound: 155.8593353
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8589362, upper bound: 155.8593833
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656379, upper bound: 155.8656447
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656445
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8610525, upper bound: 155.8608135
time: 1.09 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8668787, upper bound: 155.8669037
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8668787, upper bound: 155.8668956
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8507324, upper bound: 155.8507752
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8507324, upper bound: 155.8507752
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8564580, upper bound: 155.8565242
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8564583, upper bound: 155.8565242
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8468966, upper bound: 155.8468966
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8468966, upper bound: 155.8468966
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8569149, upper bound: 155.8569867
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8569205, upper bound: 155.8569149
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8610130, upper bound: 155.8610130
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8610274, upper bound: 155.8610130
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8658118, upper bound: 155.8658041
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8658053, upper bound: 155.8658041
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8646919, upper bound: 155.8646922
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8646919, upper bound: 155.8646925
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8662415, upper bound: 155.8662363
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8662399, upper bound: 155.8663496
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656449
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656073
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8589362, upper bound: 155.8593353
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8589362, upper bound: 155.8593833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8656379, upper bound: 155.8656447
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8656129, upper bound: 155.8656445
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.19
Output dim: 4, lower bound: -155.8610525, upper bound: 155.8608135

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8472055, upper bound: 155.8468777
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468777, upper bound: 155.8468908
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517578
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564566, upper bound: 155.8564529
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564529, upper bound: 155.8565223
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567059, upper bound: 155.8567875
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567059, upper bound: 155.8567162
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8529724, upper bound: 155.8533414
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8529724, upper bound: 155.8533376
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609582, upper bound: 155.8609582
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609582, upper bound: 155.8609582
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8499230, upper bound: 155.8499230
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8499230, upper bound: 155.8499230
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658160, upper bound: 155.8657618
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658041
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658041
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8637060, upper bound: 155.8637060
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8637141, upper bound: 155.8637060
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642670, upper bound: 155.8642728
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642670, upper bound: 155.8642728
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661707, upper bound: 155.8661707
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661707, upper bound: 155.8661736
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8660437, upper bound: 155.8661484
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8660437, upper bound: 155.8661374
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635405, upper bound: 155.8635354
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635405, upper bound: 155.8635699
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656006, upper bound: 155.8655937
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8585716
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8585716
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8588478
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8586903
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656151, upper bound: 155.8656174
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656149, upper bound: 155.8656288
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655707, upper bound: 155.8655989
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655706, upper bound: 155.8655628
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607683, upper bound: 155.8607618
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8472055, upper bound: 155.8468777
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8468777, upper bound: 155.8468908
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8608135, upper bound: 155.8608135
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8598242, upper bound: 155.8598242
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8517564, upper bound: 155.8517564
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8564566, upper bound: 155.8564529
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8564529, upper bound: 155.8565223
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8567059, upper bound: 155.8567875
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8567059, upper bound: 155.8567162
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8529724, upper bound: 155.8533414
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8529724, upper bound: 155.8533376
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8609582, upper bound: 155.8609582
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8609582, upper bound: 155.8609582
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8499230, upper bound: 155.8499230
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8499230, upper bound: 155.8499230
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8658160, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658041
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8658041, upper bound: 155.8658041
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8637060, upper bound: 155.8637060
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8637141, upper bound: 155.8637060
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8642670, upper bound: 155.8642728
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8642670, upper bound: 155.8642728
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8661707, upper bound: 155.8661707
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8661707, upper bound: 155.8661736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8660437, upper bound: 155.8661484
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8660437, upper bound: 155.8661374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8635405, upper bound: 155.8635354
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8635405, upper bound: 155.8635699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8656006, upper bound: 155.8655937
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8585716
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8585716
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8588478
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8582461, upper bound: 155.8586903
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8656151, upper bound: 155.8656174
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8656149, upper bound: 155.8656288
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8655707, upper bound: 155.8655989
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8655706, upper bound: 155.8655628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 4, lower bound: -155.8607683, upper bound: 155.8607618

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8573102, upper bound: 155.8573102
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8573102, upper bound: 155.8573102
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8601701, upper bound: 155.8601701
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8605962, upper bound: 155.8601701
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597648, upper bound: 155.8597648
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597648, upper bound: 155.8597648
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8480379, upper bound: 155.8480379
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8480379, upper bound: 155.8480379
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8563036, upper bound: 155.8562930
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8563023, upper bound: 155.8562930
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8564475
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8565127
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567044, upper bound: 155.8567044
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567044, upper bound: 155.8567856
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8471114, upper bound: 155.8472121
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8471114, upper bound: 155.8471881
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8472349
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8472349
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8468586
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8468598
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8602427, upper bound: 155.8602427
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8602427, upper bound: 155.8602427
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609238, upper bound: 155.8609238
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8609238, upper bound: 155.8609238
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658018, upper bound: 155.8657618
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8658160, upper bound: 155.8657618
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8498704, upper bound: 155.8498704
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8500978, upper bound: 155.8498704
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656097, upper bound: 155.8656073
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8656073, upper bound: 155.8656073
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636810, upper bound: 155.8636583
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636629, upper bound: 155.8636583
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566601, upper bound: 155.8566601
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566601, upper bound: 155.8566601
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642182, upper bound: 155.8642182
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642182, upper bound: 155.8642244
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8611908, upper bound: 155.8611908
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8611908, upper bound: 155.8611908
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8646247, upper bound: 155.8646247
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8646247, upper bound: 155.8646274
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659706, upper bound: 155.8660047
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659706, upper bound: 155.8660791
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659915, upper bound: 155.8660035
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659915, upper bound: 155.8661019
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635235, upper bound: 155.8635181
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635181, upper bound: 155.8635181
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8474537, upper bound: 155.8474537
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8474537, upper bound: 155.8474537
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583308
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583218
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583300
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583213
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8582021
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8588053
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8549836
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635399, upper bound: 155.8635403
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635449
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655333, upper bound: 155.8655439
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655228, upper bound: 155.8655460
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8654905, upper bound: 155.8655128
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8654901, upper bound: 155.8655164
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655700, upper bound: 155.8655628
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655628, upper bound: 155.8655628
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8454423, upper bound: 155.8454423
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8454423, upper bound: 155.8454423
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168
time: 0.83 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8573102, upper bound: 155.8573102
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8573102, upper bound: 155.8573102
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8601701, upper bound: 155.8601701
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8605962, upper bound: 155.8601701
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8597648, upper bound: 155.8597648
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8597648, upper bound: 155.8597648
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8480379, upper bound: 155.8480379
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8480379, upper bound: 155.8480379
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8563036, upper bound: 155.8562930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8563023, upper bound: 155.8562930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8564475
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8565127
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8567044, upper bound: 155.8567044
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8567044, upper bound: 155.8567856
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8471114, upper bound: 155.8472121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8471114, upper bound: 155.8471881
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8472349
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8472349
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8468586
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8468579, upper bound: 155.8468598
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8602427, upper bound: 155.8602427
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8602427, upper bound: 155.8602427
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8609238, upper bound: 155.8609238
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8609238, upper bound: 155.8609238
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8658018, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8658160, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8498704, upper bound: 155.8498704
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8500978, upper bound: 155.8498704
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8657618, upper bound: 155.8657618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8656097, upper bound: 155.8656073
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8656073, upper bound: 155.8656073
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8636810, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8636629, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8566601, upper bound: 155.8566601
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8566601, upper bound: 155.8566601
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8642182, upper bound: 155.8642182
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8642182, upper bound: 155.8642244
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8611908, upper bound: 155.8611908
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8611908, upper bound: 155.8611908
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8646247, upper bound: 155.8646247
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8646247, upper bound: 155.8646274
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8659706, upper bound: 155.8660047
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8659706, upper bound: 155.8660791
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8659915, upper bound: 155.8660035
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8659915, upper bound: 155.8661019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635235, upper bound: 155.8635181
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635181, upper bound: 155.8635181
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8474537, upper bound: 155.8474537
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8474537, upper bound: 155.8474537
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655937, upper bound: 155.8655937
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583308
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583218
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583300
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583213
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8582021
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8588053
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8549836
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635399, upper bound: 155.8635403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655333, upper bound: 155.8655439
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655228, upper bound: 155.8655460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8654905, upper bound: 155.8655128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8654901, upper bound: 155.8655164
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655700, upper bound: 155.8655628
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8655628, upper bound: 155.8655628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8454423, upper bound: 155.8454423
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8454423, upper bound: 155.8454423
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.76
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8471624, upper bound: 155.8471624
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8471624, upper bound: 155.8471624
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8561256, upper bound: 155.8561256
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8561256, upper bound: 155.8561256
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8583213, upper bound: 155.8580169
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8605301, upper bound: 155.8600843
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8600843, upper bound: 155.8600843
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597393, upper bound: 155.8597393
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8597393, upper bound: 155.8597393
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8572833, upper bound: 155.8572833
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8572833, upper bound: 155.8572833
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562930, upper bound: 155.8562930
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562930, upper bound: 155.8562930
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562836, upper bound: 155.8562836
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8562836, upper bound: 155.8562836
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517390, upper bound: 155.8517390
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517390, upper bound: 155.8517390
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8434641, upper bound: 155.8438834
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8434371, upper bound: 155.8434371
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8467303, upper bound: 155.8467303
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8467303, upper bound: 155.8467303
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8564433
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8565293
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8602100, upper bound: 155.8602100
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8602100, upper bound: 155.8602100
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564007, upper bound: 155.8564007
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564007, upper bound: 155.8564007
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8455044, upper bound: 155.8455044
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8455044, upper bound: 155.8455057
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497790, upper bound: 155.8497790
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497790, upper bound: 155.8497790
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655696, upper bound: 155.8655628
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655628, upper bound: 155.8655628
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655669, upper bound: 155.8655628
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8655726, upper bound: 155.8655628
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8634891, upper bound: 155.8634891
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8634891, upper bound: 155.8634891
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8433787, upper bound: 155.8433787
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8433787, upper bound: 155.8433787
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8437468, upper bound: 155.8434827
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8434700, upper bound: 155.8434691
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8636811, upper bound: 155.8636583
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564734, upper bound: 155.8564734
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8564734, upper bound: 155.8564734
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8641418, upper bound: 155.8641418
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8641418, upper bound: 155.8641490
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8588882, upper bound: 155.8588882
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8588882, upper bound: 155.8588882
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8520550, upper bound: 155.8520550
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8520550, upper bound: 155.8520550
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8645949, upper bound: 155.8645949
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8645949, upper bound: 155.8645949
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8641770, upper bound: 155.8641770
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8641770, upper bound: 155.8641770
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659181, upper bound: 155.8659181
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8659181, upper bound: 155.8659611
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8595157, upper bound: 155.8601121
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8595157, upper bound: 155.8601121
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642151
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642662
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8644125
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8554318, upper bound: 155.8554318
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8554318, upper bound: 155.8554318
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8473621, upper bound: 155.8473621
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8473621, upper bound: 155.8473621
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8450698, upper bound: 155.8450698
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8450698, upper bound: 155.8450698
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8579665, upper bound: 155.8579665
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8579665, upper bound: 155.8582872
time: 0.91 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8471624, upper bound: 155.8471624
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8471624, upper bound: 155.8471624
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8561256, upper bound: 155.8561256
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8561256, upper bound: 155.8561256
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8583213, upper bound: 155.8580169
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8605301, upper bound: 155.8600843
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8600843, upper bound: 155.8600843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8597393, upper bound: 155.8597393
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8597393, upper bound: 155.8597393
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8572833, upper bound: 155.8572833
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8572833, upper bound: 155.8572833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8562930, upper bound: 155.8562930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8562930, upper bound: 155.8562930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8562836, upper bound: 155.8562836
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8562836, upper bound: 155.8562836
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8517390, upper bound: 155.8517390
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8517390, upper bound: 155.8517390
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8434641, upper bound: 155.8438834
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8434371, upper bound: 155.8434371
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8467303, upper bound: 155.8467303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8467303, upper bound: 155.8467303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8564433
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564433, upper bound: 155.8565293
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8602100, upper bound: 155.8602100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8602100, upper bound: 155.8602100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564007, upper bound: 155.8564007
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564007, upper bound: 155.8564007
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8455044, upper bound: 155.8455044
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8455044, upper bound: 155.8455057
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497790, upper bound: 155.8497790
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497790, upper bound: 155.8497790
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8655696, upper bound: 155.8655628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8655628, upper bound: 155.8655628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8655669, upper bound: 155.8655628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8655726, upper bound: 155.8655628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8580169
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8634891, upper bound: 155.8634891
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8634891, upper bound: 155.8634891
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8433787, upper bound: 155.8433787
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8433787, upper bound: 155.8433787
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8437468, upper bound: 155.8434827
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8434700, upper bound: 155.8434691
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636583, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8636811, upper bound: 155.8636583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564734, upper bound: 155.8564734
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8564734, upper bound: 155.8564734
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8641418, upper bound: 155.8641418
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8641418, upper bound: 155.8641490
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8588882, upper bound: 155.8588882
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8588882, upper bound: 155.8588882
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8520550, upper bound: 155.8520550
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8520550, upper bound: 155.8520550
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8645949, upper bound: 155.8645949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8645949, upper bound: 155.8645949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8641770, upper bound: 155.8641770
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8641770, upper bound: 155.8641770
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8659181, upper bound: 155.8659181
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8659181, upper bound: 155.8659611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8595157, upper bound: 155.8601121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8595157, upper bound: 155.8601121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642045
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8642662
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8642045, upper bound: 155.8644125
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8554318, upper bound: 155.8554318
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8554318, upper bound: 155.8554318
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8473621, upper bound: 155.8473621
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8473621, upper bound: 155.8473621
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8450698, upper bound: 155.8450698
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8450698, upper bound: 155.8450698
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8586973, upper bound: 155.8586973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8497524, upper bound: 155.8497524
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8579665, upper bound: 155.8579665
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.96
Output dim: 4, lower bound: -155.8579665, upper bound: 155.8582872
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583218
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583300
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8580169, upper bound: 155.8583213
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8582021
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8582021, upper bound: 155.8588053
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8543405
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8543405, upper bound: 155.8549836
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8635399, upper bound: 155.8635403
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8635354, upper bound: 155.8635449
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8655333, upper bound: 155.8655439
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8655228, upper bound: 155.8655460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8654905, upper bound: 155.8655128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8654901, upper bound: 155.8655164
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8655700, upper bound: 155.8655628
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8655628, upper bound: 155.8655628
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607618, upper bound: 155.8607618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.96
Output dim: 4, lower bound: -155.8607168, upper bound: 155.8607168

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.37 + 417.84 = 421.20 seconds
