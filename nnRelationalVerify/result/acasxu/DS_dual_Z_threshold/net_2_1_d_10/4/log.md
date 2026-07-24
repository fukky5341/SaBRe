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
execution time: IAR + RelationalAnalysis = 1.19 + 2.28 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -155.8676494, upper bound: 155.8676494

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8676381, upper bound: 155.8676381
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8676381, upper bound: 155.8676381
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 4, lower bound: -155.8676381, upper bound: 155.8676381
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 4, lower bound: -155.8676381, upper bound: 155.8676381

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8661711
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8663273
time: 1.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8661711
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8663273
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.74 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8661711
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8663273
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8661711
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -155.8661711, upper bound: 155.8663273

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566532, upper bound: 155.8566604
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566532, upper bound: 155.8566604
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566572, upper bound: 155.8567250
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566572, upper bound: 155.8567250
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566572
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8566572
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566532
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566532
time: 1.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566532, upper bound: 155.8566604
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566532, upper bound: 155.8566604
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566572, upper bound: 155.8567250
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566572, upper bound: 155.8567250
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566572
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8567250, upper bound: 155.8566572
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566532
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 4, lower bound: -155.8566604, upper bound: 155.8566532

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566494
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566494
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566468, upper bound: 155.8566437
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566468, upper bound: 155.8566437
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566477, upper bound: 155.8567155
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566477
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566468
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8567155, upper bound: 155.8566477
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566468
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566509, upper bound: 155.8566437
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566494
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566494
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566468, upper bound: 155.8566437
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566509
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566468, upper bound: 155.8566437
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566477, upper bound: 155.8567155
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566477
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566468
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8567155, upper bound: 155.8566477
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566437, upper bound: 155.8566468
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566509, upper bound: 155.8566437
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.36
Output dim: 4, lower bound: -155.8566494, upper bound: 155.8566437

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565333, upper bound: 155.8565226
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565306, upper bound: 155.8565290
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565241, upper bound: 155.8565226
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565237, upper bound: 155.8565314
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565333, upper bound: 155.8565226
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565290
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565241, upper bound: 155.8565226
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565239, upper bound: 155.8565314
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565312, upper bound: 155.8565226
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565226
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565234, upper bound: 155.8565960
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565234, upper bound: 155.8565487
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565312, upper bound: 155.8565226
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565226
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565960
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565511
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565511, upper bound: 155.8565238
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565299
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565238
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565312
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565234
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565960, upper bound: 155.8565299
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565226
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565312
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565239
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565241
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565290, upper bound: 155.8565306
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565333
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565314, upper bound: 155.8565237
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565241
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565290, upper bound: 155.8565306
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565333
time: 1.03 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565333, upper bound: 155.8565226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565306, upper bound: 155.8565290
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565241, upper bound: 155.8565226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565237, upper bound: 155.8565314
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565333, upper bound: 155.8565226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565290
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565241, upper bound: 155.8565226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565239, upper bound: 155.8565314
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565312, upper bound: 155.8565226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565234, upper bound: 155.8565960
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565234, upper bound: 155.8565487
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565312, upper bound: 155.8565226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565960
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565238, upper bound: 155.8565511
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565511, upper bound: 155.8565238
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565299
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565238
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565312
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565234
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565960, upper bound: 155.8565299
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565312
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565239
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565241
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565290, upper bound: 155.8565306
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565333
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565314, upper bound: 155.8565237
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565241
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565290, upper bound: 155.8565306
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 4, lower bound: -155.8565226, upper bound: 155.8565333

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512983
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512983
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512898
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512898
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512897
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512897
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512896, upper bound: 155.8512895
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512971, upper bound: 155.8512894
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513133
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513133
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512893
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8513902
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513514
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512907
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512907
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512905
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8519876
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8514805
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512896, upper bound: 155.8523255
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8517827
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8517827, upper bound: 155.8512894
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512896
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512893
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8519876, upper bound: 155.8512895
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512895
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512907, upper bound: 155.8512891
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512907, upper bound: 155.8512893
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512891
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8512894
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512891
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512895
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8512971
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513042
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513387
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513387
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512905
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512905
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8512891
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8513016
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512898, upper bound: 155.8512891
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512898, upper bound: 155.8512895
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8515427
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8515240
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512983, upper bound: 155.8523229
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8523166
time: 0.88 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512983
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512898
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512898
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512897
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512897
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512896, upper bound: 155.8512895
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512971, upper bound: 155.8512894
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513133
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513133
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512893
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8513902
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513514
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512907
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512907
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512905
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512891
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8519876
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8514805
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512896, upper bound: 155.8523255
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8517827
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8517827, upper bound: 155.8512894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8512896
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512893
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8519876, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512905, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512907, upper bound: 155.8512891
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512907, upper bound: 155.8512893
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512891
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8512894
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512891
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512893, upper bound: 155.8512895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512894, upper bound: 155.8512971
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8513042
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512891
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513387
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8513387
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512905
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8512905
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8512891
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512897, upper bound: 155.8513016
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512898, upper bound: 155.8512891
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512898, upper bound: 155.8512895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8515427
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512895, upper bound: 155.8515240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512983, upper bound: 155.8523229
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 4, lower bound: -155.8512891, upper bound: 155.8523166

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8523036
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522756
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8523011
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522763
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522949
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522675
time: 0.95 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.16 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8523036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8523011
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522763
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522949
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 4, lower bound: -155.8512708, upper bound: 155.8522675

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8426632
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8421978
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8427409
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422007
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8428535
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422009
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8428829
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422125
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8431601
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8432267
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -81.3169098, 96.3277130, -81.3169098, 96.3277130, -177.6446228, 177.6446228
1: -62.8572426, 77.7227859, -62.8572426, 77.7227859, -140.5800323, 140.5800323
2: -55.0245743, 76.3968430, -55.0245743, 76.3968430, -131.4214172, 131.4214172
3: -72.9154510, 93.5727158, -72.9154510, 93.5727158, -166.4881134, 166.4881134
4: -72.2197571, 103.2277756, -72.2197571, 103.2277756, -175.4475403, 175.4475403

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8431557
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8432262
time: 0.92 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.34 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8426632
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8421978
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8427409
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422007
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8428535
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422009
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8428829
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8422125
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8431601
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8432267
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8431557
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.34
Output dim: 4, lower bound: -155.8421951, upper bound: 155.8432262

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.47 + 230.93 = 234.40 seconds
