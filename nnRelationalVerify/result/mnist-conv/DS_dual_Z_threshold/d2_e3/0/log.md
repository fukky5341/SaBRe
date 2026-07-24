## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.8648028314999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8557234, 1.8557234)
1: (-9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7110176, 1.7110167)
2: (-7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4690595, 1.4690595)
3: (-5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9553661, 1.9553661)
4: (-9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6807356, 1.6807356)
5: (1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1920395, 1.1920395)
6: (-1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.4119353, 1.4119353)
7: (-10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3383250, 1.3383250)
8: (5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895)
9: (-5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2933002, 1.2932997)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.72 + 35.24 = 57.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8656685, upper bound: 0.8656684

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653805, upper bound: 0.8656637
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653804, upper bound: 0.8653789
time: 4.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.02 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 8, lower bound: -0.8653805, upper bound: 0.8656637
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.02
Output dim: 8, lower bound: -0.8653804, upper bound: 0.8653789

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8552570, 1.8563547
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7109509, 1.7111073
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4677701, 1.4708161
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9564829, 1.9545460
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6813059, 1.6803131
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1933279, 1.1910939
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.4118381, 1.4120669
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3393455, 1.3375764
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2937498, 1.2929711

Time for backsubstitution: 20.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652960, upper bound: 0.8656639
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653795, upper bound: 0.8655803
time: 5.20 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8557234, 1.8552570
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7110176, 1.7109509
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4690595, 1.4677701
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9545460, 1.9553661
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6803131, 1.6807356
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1910939, 1.1920395
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.4119353, 1.4118376
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3375764, 1.3383250
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2929716, 1.2932997

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655804, upper bound: 0.8653795
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656639, upper bound: 0.8652959
time: 4.86 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.72
Output dim: 8, lower bound: -0.8652960, upper bound: 0.8656639
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.72
Output dim: 8, lower bound: -0.8653795, upper bound: 0.8655803
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.72
Output dim: 8, lower bound: -0.8655804, upper bound: 0.8653795
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.72
Output dim: 8, lower bound: -0.8656639, upper bound: 0.8652959

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8524694, 1.8530130
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7191563, 1.7211013
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4652166, 1.4686890
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9539242, 1.9528542
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6795177, 1.6788216
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1995716, 1.1963496
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970504, 1.3941536
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3119740, 1.3147464
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2957954, 1.2953863

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8648934, upper bound: 0.8656625
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652946, upper bound: 0.8652613
time: 5.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8519154, 1.8535662
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7209454, 1.7193131
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4656429, 1.4682622
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9547911, 1.9519873
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6798134, 1.6785250
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1985836, 1.1973376
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3939242, 1.3972797
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3165154, 1.3102050
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2961636, 1.2950177

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649770, upper bound: 0.8655790
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653782, upper bound: 0.8651777
time: 4.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529358, 1.8519154
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7192249, 1.7209449
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4665065, 1.4656429
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9519873, 1.9536743
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6785250, 1.6792440
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1973376, 1.1972952
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971467, 1.3939242
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3102055, 1.3154960
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2950182, 1.2957144

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8651778, upper bound: 0.8653781
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655790, upper bound: 0.8649769
time: 4.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523817, 1.8524694
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7210121, 1.7191567
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4669337, 1.4652162
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528542, 1.9528074
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788216, 1.6789474
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1963496, 1.1982832
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3940206, 1.3970509
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3147469, 1.3109546
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2953863, 1.2953463

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652614, upper bound: 0.8652946
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656626, upper bound: 0.8648935
time: 4.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8648934, upper bound: 0.8656625
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8652946, upper bound: 0.8652613
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8649770, upper bound: 0.8655790
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8653782, upper bound: 0.8651777
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8651778, upper bound: 0.8653781
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8655790, upper bound: 0.8649769
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8652614, upper bound: 0.8652946
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 8, lower bound: -0.8656626, upper bound: 0.8648935

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8522034, 1.8529539
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7145348, 1.7200341
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4631572, 1.4682136
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9522457, 1.9455547
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6750317, 1.6777887
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1993933, 1.1963129
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3969421, 1.3941340
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3103328, 1.3143673
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2924156, 1.2946076

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8648858, upper bound: 0.8630866
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8623162, upper bound: 0.8656552
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8524113, 1.8527460
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7180901, 1.7164788
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4647412, 1.4666300
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9466248, 1.9511757
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6784840, 1.6743355
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1995349, 1.1961713
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970313, 1.3940449
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3115945, 1.3131051
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2950172, 1.2920065

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652870, upper bound: 0.8626840
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8627189, upper bound: 0.8652542
time: 4.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8516493, 1.8535080
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7163219, 1.7182469
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4635835, 1.4677868
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9531126, 1.9446878
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6753273, 1.6774921
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1984053, 1.1973004
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3938160, 1.3972602
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3148737, 1.3098259
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2927837, 1.2942395

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649693, upper bound: 0.8630036
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8623999, upper bound: 0.8655716
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8518572, 1.8533001
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7198772, 1.7146916
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4651675, 1.4662027
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9474916, 1.9503088
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6787806, 1.6740389
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1985469, 1.1971588
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3939047, 1.3971715
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3161359, 1.3085637
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2953854, 1.2916384

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653705, upper bound: 0.8626004
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8628025, upper bound: 0.8651704
time: 4.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8526678, 1.8518572
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7146025, 1.7198777
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4644465, 1.4651675
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9503088, 1.9463749
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6740389, 1.6782112
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1971588, 1.1972585
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970380, 1.3939052
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3085642, 1.3151169
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2916374, 1.2949362

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8651704, upper bound: 0.8628025
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8626004, upper bound: 0.8653705
time: 4.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8528757, 1.8516493
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7181578, 1.7163224
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4660306, 1.4635839
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9446878, 1.9519968
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6774921, 1.6747580
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1973004, 1.1971169
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971272, 1.3938155
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3098259, 1.3138547
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2942390, 1.2923346

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655715, upper bound: 0.8623999
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630030, upper bound: 0.8649695
time: 4.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8521137, 1.8524103
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7163906, 1.7180896
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4648728, 1.4647408
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9511757, 1.9455080
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6743355, 1.6779146
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1961713, 1.1982460
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3939114, 1.3970313
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3131051, 1.3105755
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2920055, 1.2945676

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652539, upper bound: 0.8627188
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8626840, upper bound: 0.8652871
time: 4.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523216, 1.8522034
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7199450, 1.7145352
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4664569, 1.4631567
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9455547, 1.9511299
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6777878, 1.6744614
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1963129, 1.1981044
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3940010, 1.3969421
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3143673, 1.3093133
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2946072, 1.2919664

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656551, upper bound: 0.8623163
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630866, upper bound: 0.8648868
time: 4.59 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8648858, upper bound: 0.8630866
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8623162, upper bound: 0.8656552
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8652870, upper bound: 0.8626840
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8627189, upper bound: 0.8652542
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8649693, upper bound: 0.8630036
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8623999, upper bound: 0.8655716
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8653705, upper bound: 0.8626004
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8628025, upper bound: 0.8651704
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8651704, upper bound: 0.8628025
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8626004, upper bound: 0.8653705
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8655715, upper bound: 0.8623999
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8630030, upper bound: 0.8649695
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8652539, upper bound: 0.8627188
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8626840, upper bound: 0.8652871
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8656551, upper bound: 0.8623163
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.12
Output dim: 8, lower bound: -0.8630866, upper bound: 0.8648868

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8521967, 1.8528719
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7142868, 1.7200112
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4623351, 1.4681401
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9522295, 1.9453650
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6749668, 1.6769257
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1992640, 1.1947699
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3968210, 1.3927217
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3102369, 1.3131433
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2923570, 1.2939553

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8635879, upper bound: 0.8630852
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8648846, upper bound: 0.8617871
time: 4.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8521214, 1.8529472
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7145119, 1.7197852
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4630828, 1.4673915
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9520559, 1.9455385
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6741695, 1.6777229
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1978502, 1.1961837
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3955297, 1.3940129
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3091087, 1.3142724
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2917628, 1.2945490

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8610167, upper bound: 0.8656539
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8623150, upper bound: 0.8643573
time: 4.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8524036, 1.8526640
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7178402, 1.7164559
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4639192, 1.4665565
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9466085, 1.9509859
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6784191, 1.6734734
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1994057, 1.1946282
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3969102, 1.3926325
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3114991, 1.3118811
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2949586, 1.2913542

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8639891, upper bound: 0.8626833
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8652858, upper bound: 0.8613845
time: 4.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523283, 1.8527393
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7180672, 1.7162299
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4646668, 1.4658074
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9464350, 1.9511595
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6776218, 1.6742706
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1979918, 1.1960421
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3956189, 1.3939238
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3103704, 1.3130102
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2943645, 1.2919474

Time for backsubstitution: 21.84 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.96 + 563.39 = 621.35 seconds
