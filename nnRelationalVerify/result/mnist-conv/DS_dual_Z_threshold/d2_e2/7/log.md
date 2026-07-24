## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.259609539


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734241, 0.5734241)
1: (-4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444160, 0.5444160)
2: (-5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155604)
3: (-10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4519312, 0.4519312)
4: (4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382333, 0.5382330)
5: (-7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3759611, 0.3759613)
6: (-3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690289, 0.5690289)
7: (-6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293364, 0.6293364)
8: (-3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772696, 0.4772696)
9: (-6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227687, 0.5227687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.36 + 34.49 = 56.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2676387, upper bound: 0.2676391

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2665313
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665309, upper bound: 0.2676390
time: 3.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.01
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2665313
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.01
Output dim: 4, lower bound: -0.2665309, upper bound: 0.2676390

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734103, 0.5734129
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444179, 0.5444174
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155610, 0.4155604
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4519324, 0.4519308
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382400, 0.5382416
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3759620, 0.3759623
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690279, 0.5690279
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293354, 0.6293349
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772713, 0.4772720
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227561, 0.5227586

Time for backsubstitution: 20.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2599443, upper bound: 0.2631607
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2588368
time: 4.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734129, 0.5734103
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444174, 0.5444179
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4155610
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4519310, 0.4519324
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382414, 0.5382400
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3759625, 0.3759623
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690279, 0.5690284
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293349, 0.6293354
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772720, 0.4772716
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227587, 0.5227560

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2588363, upper bound: 0.2642678
time: 7.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2631602, upper bound: 0.2599445
time: 3.88 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.03
Output dim: 4, lower bound: -0.2599443, upper bound: 0.2631607
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.03
Output dim: 4, lower bound: -0.2642679, upper bound: 0.2588368
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.03
Output dim: 4, lower bound: -0.2588363, upper bound: 0.2642678
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.03
Output dim: 4, lower bound: -0.2631602, upper bound: 0.2599445

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5737383, 0.5738771
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5280657, 0.5281230
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4102720, 0.4105053
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4590931, 0.4573774
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5325310, 0.5420036
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3472501, 0.3388776
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5397694, 0.5478706
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6329570, 0.6339812
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4799631, 0.4794865
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5335090, 0.5365567

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1489

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2595647, upper bound: 0.2624755
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2593541, upper bound: 0.2627798
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5738745, 0.5737410
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5281236, 0.5280653
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4105057, 0.4102715
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4573791, 0.4590917
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5420020, 0.5325327
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3388773, 0.3472502
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5478709, 0.5397692
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6339817, 0.6329565
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4794860, 0.4799635
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5365541, 0.5335116

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1489

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2638862, upper bound: 0.2581538
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2636755, upper bound: 0.2584581
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5737410, 0.5738745
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5280652, 0.5281235
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4102715, 0.4105059
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4590917, 0.4573791
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5325329, 0.5420020
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3472503, 0.3388773
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5397694, 0.5478711
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6329565, 0.6339815
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4799635, 0.4794860
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5335116, 0.5365539

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1489

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2584580, upper bound: 0.2636756
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2581535, upper bound: 0.2638860
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5738773, 0.5737381
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5281231, 0.5280658
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4105052, 0.4102722
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4573774, 0.4590931
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5420039, 0.5325310
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3388776, 0.3472500
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5478709, 0.5397696
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6339812, 0.6329570
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4794865, 0.4799631
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5365567, 0.5335090

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1489

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2627795, upper bound: 0.2593545
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2624754, upper bound: 0.2595649
time: 3.81 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2595647, upper bound: 0.2624755
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2593541, upper bound: 0.2627798
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2638862, upper bound: 0.2581538
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2636755, upper bound: 0.2584581
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2584580, upper bound: 0.2636756
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2581535, upper bound: 0.2638860
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2627795, upper bound: 0.2593545
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.53
Output dim: 4, lower bound: -0.2624754, upper bound: 0.2595649

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5394738, 0.5335176
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5652893, 0.5703924
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3738304, 0.3648404
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4621551, 0.4617255
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5413265, 0.5417812
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3785102, 0.3782439
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5138664, 0.5265272
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6385684, 0.6414995
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831207, 0.4831426
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5243247, 0.5242438

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2482076, upper bound: 0.2598617
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2583121, upper bound: 0.2521353
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5335147, 0.5394766
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5703928, 0.5652888
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3648411, 0.3738297
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4617269, 0.4621537
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5417795, 0.5413280
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3782437, 0.3785102
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5265276, 0.5138662
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6415000, 0.6385684
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831419, 0.4831214
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5242412, 0.5243273

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2480004, upper bound: 0.2601663
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2581016, upper bound: 0.2524362
time: 3.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5394738, 0.5335176
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5652893, 0.5703924
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3738304, 0.3648404
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4621551, 0.4617255
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5413265, 0.5417812
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3785102, 0.3782439
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5138664, 0.5265272
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6385684, 0.6414995
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831207, 0.4831426
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5243247, 0.5242438

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2535429, upper bound: 0.2569012
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2612726, upper bound: 0.2467994
time: 3.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5335147, 0.5394766
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5703928, 0.5652888
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3648411, 0.3738297
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4617269, 0.4621537
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5417795, 0.5413280
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3782437, 0.3785102
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5265276, 0.5138662
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6415000, 0.6385684
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831419, 0.4831214
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5242412, 0.5243273

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2533360, upper bound: 0.2572058
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2610621, upper bound: 0.2471007
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5394766, 0.5335147
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5652888, 0.5703928
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3738297, 0.3648410
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4621537, 0.4617271
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5413280, 0.5417793
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3785102, 0.3782434
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5138662, 0.5265276
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6385684, 0.6415000
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831214, 0.4831419
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5243273, 0.5242412

Time for backsubstitution: 20.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2471006, upper bound: 0.2610621
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2572056, upper bound: 0.2533362
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5335176, 0.5394738
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5703924, 0.5652893
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3648404, 0.3738304
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4617255, 0.4621551
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5417814, 0.5413263
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3782437, 0.3785100
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5265272, 0.5138664
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6414995, 0.6385686
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831426, 0.4831207
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5242438, 0.5243247

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2467993, upper bound: 0.2612726
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2569011, upper bound: 0.2535431
time: 3.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5394766, 0.5335147
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5652888, 0.5703928
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3738297, 0.3648410
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4621537, 0.4617271
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5413280, 0.5417793
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3785102, 0.3782434
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5138662, 0.5265276
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6385684, 0.6415000
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831214, 0.4831419
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5243273, 0.5242412

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2524361, upper bound: 0.2581019
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601662, upper bound: 0.2480007
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5335176, 0.5394738
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5703924, 0.5652893
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.3648404, 0.3738304
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4617255, 0.4621551
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5417814, 0.5413263
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3782437, 0.3785100
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5265272, 0.5138664
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6414995, 0.6385686
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4831426, 0.4831207
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5242438, 0.5243247

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1747

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2521351, upper bound: 0.2583125
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2598618, upper bound: 0.2482079
time: 3.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2482076, upper bound: 0.2598617
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2583121, upper bound: 0.2521353
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2480004, upper bound: 0.2601663
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2581016, upper bound: 0.2524362
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2535429, upper bound: 0.2569012
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2612726, upper bound: 0.2467994
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2533360, upper bound: 0.2572058
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2610621, upper bound: 0.2471007
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2471006, upper bound: 0.2610621
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2572056, upper bound: 0.2533362
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2467993, upper bound: 0.2612726
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2569011, upper bound: 0.2535431
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2524361, upper bound: 0.2581019
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2601662, upper bound: 0.2480007
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2521351, upper bound: 0.2583125
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 4, lower bound: -0.2598618, upper bound: 0.2482079

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5772612, 0.5809700
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5569715, 0.5541739
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4072559, 0.4061589
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4215612, 0.4205861
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.4994142, 0.5012200
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3733625, 0.3732240
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5231519, 0.5160940
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.5563073, 0.5656382
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4736037, 0.4737668
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5296621, 0.5307730

Time for backsubstitution: 20.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1489

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2408680, upper bound: 0.2545514
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2434724, upper bound: 0.2533930
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5772612, 0.5809700
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5569715, 0.5541739
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4072559, 0.4061589
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4215612, 0.4205861
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.4994142, 0.5012200
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3733625, 0.3732240
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5231519, 0.5160940
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.5563073, 0.5656382
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4736037, 0.4737668
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5296621, 0.5307730

Time for backsubstitution: 20.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1489

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2408019, upper bound: 0.2549905
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2431720, upper bound: 0.2535243
time: 4.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5809672, 0.5772638
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5541744, 0.5569711
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4061596, 0.4072552
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4205875, 0.4215598
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5012183, 0.4994159
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3732238, 0.3733628
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5160942, 0.5231516
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.5656388, 0.5563070
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4737663, 0.4736044
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5307705, 0.5296649

Time for backsubstitution: 20.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1489

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2533381, upper bound: 0.2411426
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2568852, upper bound: 0.2408264
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5809672, 0.5772638
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5541744, 0.5569711
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4061596, 0.4072552
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4205875, 0.4215598
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5012183, 0.4994159
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3732238, 0.3733628
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5160942, 0.5231516
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.5656388, 0.5563070
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4737663, 0.4736044
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5307705, 0.5296649

Time for backsubstitution: 20.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Candidate
type: DSZ, layer: 3, pos: 1489

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2532802, upper bound: 0.2415769
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2565799, upper bound: 0.2409427
time: 4.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5772641, 0.5809672
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5569711, 0.5541744
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4072552, 0.4061595
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4215596, 0.4205875
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.4994159, 0.5012183
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3733628, 0.3732238
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5231516, 0.5160942
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.5563071, 0.5656387
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4736044, 0.4737663
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5296648, 0.5307707

Time for backsubstitution: 20.25 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.86 + 561.44 = 618.29 seconds
