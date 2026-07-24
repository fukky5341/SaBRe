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
execution time: IAR + RelationalAnalysis = 24.66 + 35.19 = 59.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8656685, upper bound: 0.8656684

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655858, upper bound: 0.8656682
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655858, upper bound: 0.8655847
time: 4.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 8, lower bound: -0.8655858, upper bound: 0.8656682
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 8, lower bound: -0.8655858, upper bound: 0.8655847

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529358, 1.8523817
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7192249, 1.7210126
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4665065, 1.4669333
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528074, 1.9536743
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6789474, 1.6792440
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1982832, 1.1972952
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971467, 1.3940206
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3109546, 1.3154960
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2953463, 1.2957144

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4639

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655775, upper bound: 0.8630928
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630088, upper bound: 0.8656610
time: 4.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523817, 1.8529358
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7210121, 1.7192245
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4669337, 1.4665065
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9536743, 1.9528074
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6792440, 1.6789474
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1972952, 1.1982832
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3940206, 1.3971467
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3154960, 1.3109546
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2957144, 1.2953463

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649319, upper bound: 0.8655702
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656538, upper bound: 0.8649308
time: 5.06 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.80
Output dim: 8, lower bound: -0.8655775, upper bound: 0.8630928
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.80
Output dim: 8, lower bound: -0.8630088, upper bound: 0.8656610
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.80
Output dim: 8, lower bound: -0.8649319, upper bound: 0.8655702
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.80
Output dim: 8, lower bound: -0.8656538, upper bound: 0.8649308

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529291, 1.8522997
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189751, 1.7209892
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4656835, 1.4668589
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9527912, 1.9534836
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788816, 1.6783810
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1981535, 1.1957521
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970256, 1.3926082
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3108597, 1.3142719
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2952881, 1.2950625

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649235, upper bound: 0.8630777
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655628, upper bound: 0.8624371
time: 4.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8528538, 1.8523750
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7192001, 1.7207632
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4664321, 1.4661102
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9526167, 1.9536572
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6780844, 1.6791782
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1967397, 1.1971660
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3957343, 1.3938994
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3097305, 1.3154011
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2946939, 1.2956562

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8626058, upper bound: 0.8656598
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8626057, upper bound: 0.8652585
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523788, 1.8529329
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7209549, 1.7191763
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4668393, 1.4664278
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9537582, 1.9528723
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6792088, 1.6789236
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1972518, 1.1982336
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3940811, 1.3972278
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3153005, 1.3107219
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2957153, 1.2953467

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8647255, upper bound: 0.8655658
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8650100, upper bound: 0.8652813
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523788, 1.8529320
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7209644, 1.7191668
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4668555, 1.4664121
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9537392, 1.9528913
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6792192, 1.6789131
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1972456, 1.1982398
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3941016, 1.3972073
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3152633, 1.3107591
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2957153, 1.2953467

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643581, upper bound: 0.8642877
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656526, upper bound: 0.8642749
time: 4.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8649235, upper bound: 0.8630777
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8655628, upper bound: 0.8624371
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8626058, upper bound: 0.8656598
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8626057, upper bound: 0.8652585
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8647255, upper bound: 0.8655658
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8650100, upper bound: 0.8652813
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8643581, upper bound: 0.8642877
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.92
Output dim: 8, lower bound: -0.8656526, upper bound: 0.8642749

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529263, 1.8522968
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189188, 1.7209420
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4655900, 1.4667807
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528761, 1.9535503
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788464, 1.6783562
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1981101, 1.1957021
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970857, 1.3926888
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3106637, 1.3140388
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2952881, 1.2950630

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8646344, upper bound: 0.8630724
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649189, upper bound: 0.8627892
time: 4.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529263, 1.8522968
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189283, 1.7209325
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4656062, 1.4667649
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528570, 1.9535694
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788568, 1.6783457
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1981039, 1.1957083
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971062, 1.3926678
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3106265, 1.3140759
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2952881, 1.2950630

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655625, upper bound: 0.8624375
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655627, upper bound: 0.8624362
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8525858, 1.8523140
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7145786, 1.7196960
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4643722, 1.4656348
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9509392, 1.9463577
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6735983, 1.6781454
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1965613, 1.1971288
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3956261, 1.3938799
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3080893, 1.3150215
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2913141, 1.2948775

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 466

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8613053, upper bound: 0.8656585
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8626036, upper bound: 0.8643619
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8527927, 1.8521070
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7181339, 1.7161407
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4659562, 1.4640508
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9453173, 1.9519796
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6770515, 1.6746922
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1967030, 1.1969872
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3957152, 1.3937907
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3093514, 1.3137598
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2939157, 1.2922759

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8623172, upper bound: 0.8652540
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630030, upper bound: 0.8649695
time: 4.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8519115, 1.8535633
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7208881, 1.7192664
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4655499, 1.4681849
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9548750, 1.9520521
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6797791, 1.6785002
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1985412, 1.1972890
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3939843, 1.3973608
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3163204, 1.3099728
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2961636, 1.2950177

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8646426, upper bound: 0.8655653
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8647253, upper bound: 0.8655653
time: 5.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8523788, 1.8524656
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7209549, 1.7191091
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4668393, 1.4651389
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9529381, 1.9528723
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6787863, 1.6789236
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1963072, 1.1982336
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3940811, 1.3971314
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3145514, 1.3107219
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2953863, 1.2953467

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8646074, upper bound: 0.8652799
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8650086, upper bound: 0.8648788
time: 4.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8376751, 1.8406754
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7205181, 1.7186317
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4673052, 1.4669652
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9532719, 1.9523296
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6840868, 1.6825380
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1967850, 1.1978555
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3949833, 1.3979235
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3110728, 1.3072672
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2991157, 1.2974939

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5829

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653636, upper bound: 0.8642700
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656480, upper bound: 0.8639854
time: 4.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8646344, upper bound: 0.8630724
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8649189, upper bound: 0.8627892
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8655625, upper bound: 0.8624375
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8655627, upper bound: 0.8624362
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8613053, upper bound: 0.8656585
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8626036, upper bound: 0.8643619
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8623172, upper bound: 0.8652540
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8630030, upper bound: 0.8649695
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8646426, upper bound: 0.8655653
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8647253, upper bound: 0.8655653
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8646074, upper bound: 0.8652799
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8650086, upper bound: 0.8648788
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8653636, upper bound: 0.8642700
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.10
Output dim: 8, lower bound: -0.8656480, upper bound: 0.8639854

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529263, 1.8518305
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189188, 1.7208743
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4655900, 1.4654918
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9520550, 1.9535503
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6784258, 1.6783562
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1971650, 1.1957021
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3970857, 1.3925924
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3099151, 1.3140388
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2949595, 1.2950630

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 527

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649186, upper bound: 0.8627890
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8649188, upper bound: 0.8627888
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529234, 1.8522940
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189360, 1.7209415
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4656062, 1.4667649
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528608, 1.9535732
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788473, 1.6783381
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1981025, 1.1957073
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971181, 1.3926816
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3106246, 1.3140745
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2952819, 1.2950573

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 466
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8651599, upper bound: 0.8624362
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655611, upper bound: 0.8620325
time: 4.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8529243, 1.8522930
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7189369, 1.7209406
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4656062, 1.4667649
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9528608, 1.9535732
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6788492, 1.6783361
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1981030, 1.1957073
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3971200, 1.3926797
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3106251, 1.3140740
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2952828, 1.2950563

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 5829
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 527

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8651601, upper bound: 0.8624356
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8655613, upper bound: 0.8620325
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8403292, 1.8376122
1: -9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7140436, 1.7192497
2: -7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4649243, 1.4660845
3: -5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9503765, 1.9458895
4: -9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6771545, 1.6829414
5: 1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1961746, 1.1966667
6: -1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.3963428, 1.3947625
7: -10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3045955, 1.3108306
8: 5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895
9: -5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2934608, 1.2982779

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 5829

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8613050, upper bound: 0.8656583
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8613052, upper bound: 0.8656581
time: 4.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 21.96 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.85 + 552.94 = 612.79 seconds
