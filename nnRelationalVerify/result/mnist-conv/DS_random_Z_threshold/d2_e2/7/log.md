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
execution time: IAR + RelationalAnalysis = 24.78 + 33.20 = 57.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2676387, upper bound: 0.2676391

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2665313
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2665309, upper bound: 0.2676390
time: 3.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.46
Output dim: 4, lower bound: -0.2676386, upper bound: 0.2665313
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.46
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

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1788

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1237

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2675718, upper bound: 0.2665143
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2676219, upper bound: 0.2664644
time: 3.64 seconds

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

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1509

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2649840, upper bound: 0.2674181
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2663101, upper bound: 0.2660919
time: 3.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.13
Output dim: 4, lower bound: -0.2675718, upper bound: 0.2665143
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.13
Output dim: 4, lower bound: -0.2676219, upper bound: 0.2664644
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.13
Output dim: 4, lower bound: -0.2649840, upper bound: 0.2674181
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.13
Output dim: 4, lower bound: -0.2663101, upper bound: 0.2660919

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5719643, 0.5718760
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5419979, 0.5417678
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4118001, 0.4119494
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4473112, 0.4453607
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382209, 0.5382240
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3583415, 0.3613060
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5663090, 0.5680194
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6285450, 0.6281734
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4770353, 0.4767325
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5143845, 0.5142293

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2118

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2553747, upper bound: 0.2659327
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2669955, upper bound: 0.2558610
time: 3.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5718734, 0.5719669
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5417681, 0.5419974
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4119501, 0.4117994
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4453621, 0.4473095
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382218, 0.5382228
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3613060, 0.3583417
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5680199, 0.5663087
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6281741, 0.6285448
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4767320, 0.4770358
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5142267, 0.5143871

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 1509

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2397

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2673134, upper bound: 0.2654287
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601880, upper bound: 0.2612873
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2632337, upper bound: 0.2602131
time: 4.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5761962, 0.5765846
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5420966, 0.5423162
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4114199, 0.4114767
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4493861, 0.4488649
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5373666, 0.5375266
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3771718, 0.3769555
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5657754, 0.5663054
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6274714, 0.6273150
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4671156, 0.4672792
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5154152, 0.5160304

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1788

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2649739, upper bound: 0.2667260
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2642916, upper bound: 0.2674098
time: 4.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5765872, 0.5761936
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5423157, 0.5420971
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4114761, 0.4114205
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4488633, 0.4493876
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5375288, 0.5373647
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3769557, 0.3771715
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5663052, 0.5657756
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6273146, 0.6274719
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4672797, 0.4671152
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5160332, 0.5154124

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2397

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2660019, upper bound: 0.2651010
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2653106, upper bound: 0.2657851
time: 4.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.99 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2553747, upper bound: 0.2659327
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2669955, upper bound: 0.2558610
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2601880, upper bound: 0.2612873
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2632337, upper bound: 0.2602131
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2649739, upper bound: 0.2667260
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2642916, upper bound: 0.2674098
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2660019, upper bound: 0.2651010
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.99
Output dim: 4, lower bound: -0.2653106, upper bound: 0.2657851

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5727684, 0.5720971
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5414443, 0.5445061
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4154859, 0.4155083
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4515738, 0.4465199
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5374956, 0.5386226
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3690963, 0.3670359
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5667553, 0.5672569
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6265402, 0.6262774
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4671807, 0.4730587
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5187967, 0.5158614

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1489

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2550095, upper bound: 0.2652627
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2543068, upper bound: 0.2655673
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5720944, 0.5727713
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5445065, 0.5414438
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155090, 0.4154853
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4465213, 0.4515724
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5386209, 0.5374975
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3670359, 0.3690965
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5672569, 0.5667553
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6262779, 0.6265395
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4730580, 0.4671812
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5158589, 0.5187992

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2637524, upper bound: 0.2525212
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2636550, upper bound: 0.2526185
time: 3.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5612009, 0.5622070
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5443425, 0.5443485
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4139268, 0.4147452
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4483485, 0.4518657
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5240297, 0.5243058
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3694248, 0.3690152
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5578151, 0.5580442
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6252255, 0.6246576
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4723332, 0.4709890
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5130198, 0.5181646

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2569454, upper bound: 0.2579479
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2568481, upper bound: 0.2580453
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5622096, 0.5612035
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5443490, 0.5443420
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4147458, 0.4139260
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4518671, 0.4483469
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5243077, 0.5240319
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3690147, 0.3694253
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5580444, 0.5578146
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6246576, 0.6252253
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4709885, 0.4723339
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5181670, 0.5130224

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2555394, upper bound: 0.2568415
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2598630, upper bound: 0.2525149
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5733881, 0.5733795
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5446479, 0.5442166
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4154105, 0.4153470
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4521012, 0.4521177
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5384490, 0.5384481
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3762565, 0.3762138
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5681272, 0.5680442
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6293211, 0.6292880
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772880, 0.4772692
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5217566, 0.5214076

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2334

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2397

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2646670, upper bound: 0.2657262
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2639822, upper bound: 0.2664177
time: 4.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5733821, 0.5733855
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5442162, 0.5446484
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4153464, 0.4154112
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4521163, 0.4521027
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5384500, 0.5384474
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3762140, 0.3762562
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5680442, 0.5681276
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6292877, 0.6293216
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772696, 0.4772875
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5214105, 0.5217540

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2610481, upper bound: 0.2640694
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2609508, upper bound: 0.2641669
time: 4.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5732753, 0.5736752
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444164, 0.5440786
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4153585, 0.4158074
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4512670, 0.4515302
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5379257, 0.5377703
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3753874, 0.3755751
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690022, 0.5690598
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6294508, 0.6292722
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4771893, 0.4773867
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5226808, 0.5226985

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1489
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1509

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2597494, upper bound: 0.2607127
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2608251, upper bound: 0.2576645
time: 5.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5734129, 0.5732725
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5444174, 0.5444169
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4155605, 0.4153593
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4515288, 0.4519324
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5382414, 0.5379238
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3755753, 0.3759623
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5690279, 0.5690024
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6292720, 0.6293354
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4772720, 0.4771888
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5227010, 0.5227560

Time for backsubstitution: 23.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 1489

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2371

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2590594, upper bound: 0.2613969
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2601338, upper bound: 0.2583474
time: 4.06 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2550095, upper bound: 0.2652627
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2543068, upper bound: 0.2655673
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2637524, upper bound: 0.2525212
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2636550, upper bound: 0.2526185
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2569454, upper bound: 0.2579479
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2568481, upper bound: 0.2580453
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2555394, upper bound: 0.2568415
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2598630, upper bound: 0.2525149
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2646670, upper bound: 0.2657262
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2639822, upper bound: 0.2664177
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2610481, upper bound: 0.2640694
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2609508, upper bound: 0.2641669
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2597494, upper bound: 0.2607127
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2608251, upper bound: 0.2576645
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2590594, upper bound: 0.2613969
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.22
Output dim: 4, lower bound: -0.2601338, upper bound: 0.2583474

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 2146
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1509

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2534604, upper bound: 0.2650421
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2547887, upper bound: 0.2637154
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 3109
type: DSZ, layer: 3, pos: 1166
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2397
type: DSZ, layer: 3, pos: 1237
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1747
type: DSZ, layer: 3, pos: 2146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1158

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528476, upper bound: 0.2635403
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2522804, upper bound: 0.2641102
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.1057286, -6.2920437, -7.1057286, -6.2920437, -0.5733936, 0.5733144
1: -4.9336252, -4.0876632, -4.9336252, -4.0876632, -0.5368154, 0.5393260
2: -5.6065383, -4.8446698, -5.6065383, -4.8446698, -0.4103441, 0.4083905
3: -10.7152185, -9.9662857, -10.7152185, -9.9662857, -0.4402380, 0.4423628
4: 4.3862057, 5.0951324, 4.3862057, 5.0951324, -0.5294702, 0.5280616
5: -7.9719667, -7.2899966, -7.9719667, -7.2899966, -0.3683903, 0.3675449
6: -3.2702131, -2.2950277, -3.2702131, -2.2950277, -0.5692725, 0.5693913
7: -6.3882132, -5.3162689, -6.3882132, -5.3162689, -0.6127133, 0.6180038
8: -3.3521690, -2.5929775, -3.3521690, -2.5929775, -0.4733777, 0.4729373
9: -6.7731895, -5.9340916, -6.7731895, -5.9340916, -0.5166740, 0.5165929

Time for backsubstitution: 22.60 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.98 + 543.27 = 601.25 seconds
