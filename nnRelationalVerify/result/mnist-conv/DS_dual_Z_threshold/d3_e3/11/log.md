## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.441476739


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5637841, 1.5637841)
1: (-11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1966600, 1.1966598)
2: (-7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9311540, 0.9311540)
3: (-7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4207768, 1.4207768)
4: (-3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4014397, 1.4014397)
5: (-5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2628298, 1.2628298)
6: (-16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2654819, 1.2654817)
7: (-4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3419938, 1.3419938)
8: (-4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9227247, 0.9227247)
9: (4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7659516, 0.7659515)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.09 + 34.75 = 57.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.4459361, upper bound: 0.4459358

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447841, upper bound: 0.4459336
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4447839
time: 5.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.31 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.31
Output dim: 9, lower bound: -0.4447841, upper bound: 0.4459336
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.31
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4447839

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5558972, 1.5522215
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1948833, 1.1945312
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9355547, 0.9348565
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4058704, 1.4083428
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4023132, 1.4025965
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2652516, 1.2648656
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2657714, 1.2634449
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3387327, 1.3380885
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9156635, 0.9142449
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7592785, 0.7604959

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447832, upper bound: 0.4454255
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4442738, upper bound: 0.4459350
time: 3.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5522213, 1.5558972
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1945310, 1.1948833
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9348564, 0.9355547
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4083428, 1.4058704
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4025965, 1.4023130
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2648654, 1.2652514
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2634449, 1.2657714
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3380885, 1.3387327
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9142449, 0.9156635
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7604959, 0.7592785

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459329, upper bound: 0.4442754
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4454235, upper bound: 0.4447852
time: 3.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.40 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.40
Output dim: 9, lower bound: -0.4447832, upper bound: 0.4454255
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.40
Output dim: 9, lower bound: -0.4442738, upper bound: 0.4459350
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.40
Output dim: 9, lower bound: -0.4459329, upper bound: 0.4442754
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.40
Output dim: 9, lower bound: -0.4454235, upper bound: 0.4447852

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5195651, 1.5059171
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1851377, 1.1807485
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8948882, 0.9011021
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3549271, 1.3471808
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4080086, 1.4107490
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2302637, 1.2228377
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2581272, 1.2536645
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3351388, 1.3350945
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9160106, 0.9151567
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7594273, 0.7606789

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437002, upper bound: 0.4454228
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447828, upper bound: 0.4443396
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5095925, 1.5158892
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1811008, 1.1847854
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9018002, 0.8941901
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3447084, 1.3573995
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4104652, 1.4082923
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2232237, 1.2298777
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2559910, 1.2558005
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3357382, 1.3344948
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9165752, 0.9145921
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7594616, 0.7606448

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431903, upper bound: 0.4459346
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4442734, upper bound: 0.4448519
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5158892, 1.5095925
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1847858, 1.1811006
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8941901, 0.9018002
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3573995, 1.3447084
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4082923, 1.4104652
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2298775, 1.2232234
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2558002, 1.2559910
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3344946, 1.3357387
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9145920, 0.9165752
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7606447, 0.7594615

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448499, upper bound: 0.4442753
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459325, upper bound: 0.4431897
time: 4.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5059171, 1.5195651
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1807489, 1.1851375
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9011021, 0.8948882
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3471808, 1.3549271
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4107490, 1.4080086
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2228374, 1.2302635
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2536645, 1.2581270
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3350945, 1.3351390
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9151566, 0.9160107
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7606790, 0.7594273

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443401, upper bound: 0.4447849
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4454231, upper bound: 0.4437018
time: 3.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4437002, upper bound: 0.4454228
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4447828, upper bound: 0.4443396
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4431903, upper bound: 0.4459346
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4442734, upper bound: 0.4448519
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4448499, upper bound: 0.4442753
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4459325, upper bound: 0.4431897
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4443401, upper bound: 0.4447849
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 9, lower bound: -0.4454231, upper bound: 0.4437018

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5151448, 1.5006237
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1887217, 1.1850607
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8953626, 0.9016727
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3384972, 1.3334868
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3970461, 1.3975954
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2303996, 1.2196238
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2432551, 1.2358263
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3065214, 1.3112469
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9051385, 0.9070828
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7604280, 0.7633147

Time for backsubstitution: 23.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436987, upper bound: 0.4454242
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436974, upper bound: 0.4443450
time: 3.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5142722, 1.5014968
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1894493, 1.1843324
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8954589, 0.9015764
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3412333, 1.3307507
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3948555, 1.3997865
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2270494, 1.2229741
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2402883, 1.2387929
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3112917, 1.3064764
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9079368, 0.9042844
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7620630, 0.7616795

Time for backsubstitution: 23.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437032, upper bound: 0.4443392
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447820, upper bound: 0.4443405
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5051727, 1.5105963
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1846848, 1.1890974
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9022746, 0.8947608
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3282785, 1.3437054
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3995028, 1.3951387
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2233596, 1.2266638
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2411194, 1.2379620
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3071203, 1.3106472
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9057031, 0.9065183
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7604623, 0.7632806

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431888, upper bound: 0.4459337
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431875, upper bound: 0.4448549
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5042996, 1.5114694
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1854124, 1.1883693
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9023709, 0.8946644
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3310146, 1.3409693
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3973122, 1.3973298
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2200093, 1.2300141
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2381525, 1.2409289
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3118916, 1.3058770
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9085014, 0.9037198
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7620974, 0.7616454

Time for backsubstitution: 23.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431933, upper bound: 0.4448490
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4442725, upper bound: 0.4448499
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5114694, 1.5042996
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1883688, 1.1854129
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8946643, 0.9023709
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3409691, 1.3310144
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3973298, 1.3973122
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2300134, 1.2200096
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2409291, 1.2381527
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3058767, 1.3118911
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9037199, 0.9085014
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7616453, 0.7620974

Time for backsubstitution: 23.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448484, upper bound: 0.4442745
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448471, upper bound: 0.4431953
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5105963, 1.5051727
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1890974, 1.1846845
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8947608, 0.9022745
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3437061, 1.3282783
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3951387, 1.3995030
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2266641, 1.2233598
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2379622, 1.2411194
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3106470, 1.3071206
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9065185, 0.9057029
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7632806, 0.7604622

Time for backsubstitution: 23.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448529, upper bound: 0.4431894
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459317, upper bound: 0.4431908
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5014968, 1.5142717
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1843319, 1.1894495
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9015765, 0.8954589
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3307505, 1.3412330
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3997865, 1.3948553
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2229733, 1.2270496
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2387929, 1.2402885
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3064766, 1.3112915
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9042845, 0.9079368
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7616796, 0.7620631

Time for backsubstitution: 23.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443386, upper bound: 0.4447840
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443372, upper bound: 0.4437051
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5006237, 1.5151453
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1850605, 1.1887212
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9016728, 0.8953626
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3334866, 1.3384969
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3975954, 1.3970463
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2196240, 1.2303998
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2358260, 1.2432554
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3112469, 1.3065212
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9070828, 0.9051384
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7633147, 0.7604280

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443430, upper bound: 0.4436993
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4454223, upper bound: 0.4437006
time: 3.88 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4436987, upper bound: 0.4454242
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4436974, upper bound: 0.4443450
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4437032, upper bound: 0.4443392
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4447820, upper bound: 0.4443405
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4431888, upper bound: 0.4459337
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4431875, upper bound: 0.4448549
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4431933, upper bound: 0.4448490
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4442725, upper bound: 0.4448499
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4448484, upper bound: 0.4442745
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4448471, upper bound: 0.4431953
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4448529, upper bound: 0.4431894
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4459317, upper bound: 0.4431908
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4443386, upper bound: 0.4447840
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4443372, upper bound: 0.4437051
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4443430, upper bound: 0.4436993
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.43
Output dim: 9, lower bound: -0.4454223, upper bound: 0.4437006

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5143070, 1.4993196
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1891451, 1.1856167
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8957245, 0.9021477
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3336902, 1.3294823
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3940535, 1.3940029
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2350965, 1.2232003
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2378135, 1.2293005
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.2978358, 1.3040125
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9080024, 0.9108387
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7618167, 0.7651346

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436975, upper bound: 0.4450663
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433408, upper bound: 0.4454230
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5138407, 1.4997768
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1892772, 1.1854842
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8958375, 0.9020348
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3344903, 1.3286803
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3934536, 1.3946028
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2339764, 1.2243207
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2367296, 1.2303753
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.2992864, 1.3025620
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9088941, 0.9099468
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7622468, 0.7647034

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436962, upper bound: 0.4439847
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433394, upper bound: 0.4443436
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5134249, 1.5001926
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1898732, 1.1848886
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8958209, 0.9020513
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3364263, 1.3267443
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3918629, 1.3961940
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2317462, 1.2265503
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2348375, 1.2322674
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3026061, 1.2992418
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9108007, 0.9080403
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7634518, 0.7634984

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 4670

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437020, upper bound: 0.4439812
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433453, upper bound: 0.4443360
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5129681, 1.5006590
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1900058, 1.1847560
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8959339, 0.9019383
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3372283, 1.3259442
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3912625, 1.3967938
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2306261, 1.2276707
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2337627, 1.2333512
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3040566, 1.2977917
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9116926, 0.9071484
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7638831, 0.7630682

Time for backsubstitution: 23.54 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.84 + 557.35 = 615.19 seconds
