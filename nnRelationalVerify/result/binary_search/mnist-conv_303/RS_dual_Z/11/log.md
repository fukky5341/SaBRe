## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.8650754865
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.6260853, -9.4460907, -12.6260853, -9.4460907, -3.1799946, 3.1799946)
1: (-11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.4480700, 2.4480703)
2: (-8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9626331, 1.9626331)
3: (-7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.6083369, 2.6083369)
4: (-3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864)
5: (-5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222)
6: (-16.9029446, -13.7977066, -16.9029446, -13.7977066, -3.0488920, 3.0488920)
7: (-4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329)
8: (-5.2317653, -2.9253664, -5.2317653, -2.9253664, -2.1087542, 2.1087539)
9: (4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.5658846, 1.5658846)

## BASE Result
execution time: IAR + LP analysis = 13.64 + 33.32 = 46.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.2026377, upper bound: 1.2026362


# Binary Search by BASE starts (time budget: 3553.04 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.1616536378860474
rel_dist={9: [-0.6705902224208895, 0.6705927823695186]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.2096967697143555
rel_dist={9: [-0.7384599414700519, 0.738462472703862]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=1.2577396631240845
rel_dist={9: [-0.803270111495487, 0.8032729720962895]}

## Binary Search Result
Binary search time: 194.34 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 3358.70 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412414, upper bound: 1.0414121
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414120, upper bound: 1.0412416
time: 3.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.01
Output dim: 9, lower bound: -1.0412414, upper bound: 1.0414121
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.01
Output dim: 9, lower bound: -1.0414120, upper bound: 1.0412416

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9560399, 2.9802351
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2060075, 2.2163560
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9524739, 1.9477265
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4481535, 2.4512563
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6516533, 2.6622794
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8764336, 1.8583078
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4502454, 1.4498769

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380715, upper bound: 1.0414081
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412349, upper bound: 1.0382450
time: 3.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9584851, 2.9560399
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2070518, 2.2060072
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9477265, 1.9482064
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4484658, 2.4481537
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6527281, 2.6516531
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8583081, 1.8601403
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4498768, 1.4499115

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382425, upper bound: 1.0412368
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414056, upper bound: 1.0380737
time: 3.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.73
Output dim: 9, lower bound: -1.0380715, upper bound: 1.0414081
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.73
Output dim: 9, lower bound: -1.0412349, upper bound: 1.0382450
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.73
Output dim: 9, lower bound: -1.0382425, upper bound: 1.0412368
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.73
Output dim: 9, lower bound: -1.0414056, upper bound: 1.0380737

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9494700, 2.9626844
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2038097, 2.2128513
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9626331, 1.9541926
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4280353, 2.4388781
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6517298, 2.6559155
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8728139, 1.8525085
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4423020, 1.4456985

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380693, upper bound: 1.0408048
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0374668, upper bound: 1.0414058
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9384894, 2.9736669
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2025023, 2.2141590
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9589398, 1.9584057
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4357753, 2.4311383
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6452892, 2.6623516
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8706338, 1.8546884
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4460671, 1.4419332

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412319, upper bound: 1.0376417
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0406299, upper bound: 1.0382426
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9519153, 2.9384890
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2048540, 2.2025025
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9584057, 1.9546723
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4283476, 2.4357755
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6528046, 2.6452892
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8546884, 1.8543410
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4419334, 1.4457328

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382402, upper bound: 1.0406323
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376394, upper bound: 1.0412343
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9409347, 2.9494696
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2035475, 2.2038095
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9541929, 1.9588854
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4360876, 2.4280357
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6463645, 2.6517296
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8525083, 1.8565211
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4456985, 1.4419675

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414033, upper bound: 1.0374693
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0408024, upper bound: 1.0380713
time: 3.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0380693, upper bound: 1.0408048
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0374668, upper bound: 1.0414058
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0412319, upper bound: 1.0376417
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0406299, upper bound: 1.0382426
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0382402, upper bound: 1.0406323
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0376394, upper bound: 1.0412343
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0414033, upper bound: 1.0374693
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.58
Output dim: 9, lower bound: -1.0408024, upper bound: 1.0380713

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9270263, 2.9268479
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1989117, 2.2050331
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9023149, 1.9171011
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3781552, 2.3541670
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6667361, 2.6570132
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8713932, 1.8502524
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4425075, 1.4460189

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0358321, upper bound: 1.0408037
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380681, upper bound: 1.0385689
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9136329, 2.9402409
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1959906, 2.2079537
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9260619, 1.8933542
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3433242, 2.3889976
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6528273, 2.6709244
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8705573, 1.8510880
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4426224, 1.4459040

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0352309, upper bound: 1.0414046
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0374657, upper bound: 1.0391689
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9160457, 2.9378304
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1976042, 2.2063408
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8981016, 1.9213142
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3858948, 2.3464270
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6602960, 2.6634493
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8692141, 1.8524323
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4462726, 1.4422538

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0389952, upper bound: 1.0376408
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412308, upper bound: 1.0354059
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9026523, 2.9512234
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1946840, 2.2092612
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9218485, 1.8975673
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3510642, 2.3812575
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6463871, 2.6773605
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8683777, 1.8532681
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4463878, 1.4421387

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0383940, upper bound: 1.0382415
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0406288, upper bound: 1.0360058
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9294715, 2.9026527
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1999564, 2.1946843
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8975670, 1.9175808
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3784666, 2.3510642
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6678104, 2.6463869
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8532677, 1.8520851
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4421389, 1.4460530

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0360034, upper bound: 1.0406312
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382391, upper bound: 1.0383964
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9160781, 2.9160454
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1970353, 2.1976047
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9213145, 1.8938339
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3436360, 2.3858948
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6539016, 2.6602960
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8524323, 1.8529208
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4422538, 1.4459381

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0354035, upper bound: 1.0412333
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376383, upper bound: 1.0389977
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9184909, 2.9136331
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1986489, 2.1959913
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8933542, 1.9217937
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3862066, 2.3433242
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6613703, 2.6528273
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8510880, 1.8542652
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4459040, 1.4422879

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0391664, upper bound: 1.0374682
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414022, upper bound: 1.0352334
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9050984, 2.9270258
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1957288, 2.1989117
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9171011, 1.8980470
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3513761, 2.3781550
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6474614, 2.6667361
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8502522, 1.8551006
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4460189, 1.4421730

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0385665, upper bound: 1.0380702
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0408013, upper bound: 1.0358345
time: 3.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0358321, upper bound: 1.0408037
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0380681, upper bound: 1.0385689
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0352309, upper bound: 1.0414046
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0374657, upper bound: 1.0391689
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0389952, upper bound: 1.0376408
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0412308, upper bound: 1.0354059
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0383940, upper bound: 1.0382415
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0406288, upper bound: 1.0360058
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0360034, upper bound: 1.0406312
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0382391, upper bound: 1.0383964
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0354035, upper bound: 1.0412333
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0376383, upper bound: 1.0389977
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0391664, upper bound: 1.0374682
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0414022, upper bound: 1.0352334
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0385665, upper bound: 1.0380702
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.62
Output dim: 9, lower bound: -1.0408013, upper bound: 1.0358345

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9231424, 2.9199083
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2026963, 2.2107368
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9030178, 1.9181592
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3538995, 2.3390000
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1254382
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6535707, 2.6359787
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4171629, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8557043, 1.8426201
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4411414, 1.4492654

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0358216, upper bound: 1.0403566
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0353842, upper bound: 1.0407933
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9200859, 2.9229650
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2046151, 2.2088175
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9033735, 1.9178035
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3629880, 2.3299117
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6457019, 2.6438479
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4235957
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8637609, 1.8345640
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4457541, 1.4446529

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380579, upper bound: 1.0381218
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376199, upper bound: 1.0385584
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9097500, 2.9333010
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1997752, 2.2136571
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9267642, 1.8944125
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3190689, 2.3738308
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6396618, 2.6498899
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4087200, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8548689, 1.8434558
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4412565, 1.4491504

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0352204, upper bound: 1.0409567
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0347838, upper bound: 1.0413941
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9066935, 2.9363580
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2016950, 2.2117381
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9271200, 1.8940568
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3281574, 2.3647423
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1242919, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6317925, 2.6577590
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4244461, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8629255, 1.8353996
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4458692, 1.4445380

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0374552, upper bound: 1.0387210
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370186, upper bound: 1.0391584
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9121618, 2.9308908
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2013888, 2.2120442
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8988044, 1.9223723
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3616395, 2.3312602
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6471305, 2.6424150
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4151306, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8535252, 1.8448002
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4449067, 1.4455003

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0389847, upper bound: 1.0371936
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0385473, upper bound: 1.0376303
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9091063, 2.9339476
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2033076, 2.2101252
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8991601, 1.9220166
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3707280, 2.3221717
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6392612, 2.6502838
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4256294
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8615818, 1.8367441
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4495194, 1.4408878

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412206, upper bound: 1.0349588
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0407831, upper bound: 1.0353954
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8987694, 2.9442835
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1984687, 2.2149649
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9225509, 1.8986256
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3268089, 2.3660908
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6332211, 2.6563261
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4066877, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8526897, 1.8456359
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4450219, 1.4453853

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0383835, upper bound: 1.0377936
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0379468, upper bound: 1.0382310
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8957129, 2.9473403
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2003875, 2.2130458
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9229066, 1.8982697
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3358974, 2.3570023
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1225028, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6253524, 2.6641951
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4224138, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8607454, 1.8375795
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4496343, 1.4407729

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0406183, upper bound: 1.0355579
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401817, upper bound: 1.0359953
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9255877, 2.8957129
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2037406, 2.2003877
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8982704, 1.9186389
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3542118, 2.3358974
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1225033
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6546454, 2.6253524
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4188652, 2.4224138
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8375797, 1.8444529
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4407728, 1.4492997

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0359929, upper bound: 1.0401841
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0355555, upper bound: 1.0406207
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9225321, 2.8987696
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2056594, 2.1984687
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8986261, 1.9182832
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3633003, 2.3268089
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6467767, 2.6332214
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4066877
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8456354, 1.8363967
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4453852, 1.4446870

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382286, upper bound: 1.0379493
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0377912, upper bound: 1.0383859
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9121962, 2.9091058
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2008195, 2.2033081
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9220169, 1.8948922
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3193808, 2.3707280
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6407366, 2.6392615
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4104223, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8367443, 1.8452885
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4408877, 1.4491845

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0353930, upper bound: 1.0407854
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0349564, upper bound: 1.0412228
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9091387, 2.9121623
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2027392, 2.2013893
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9223726, 1.8945365
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3284693, 2.3616395
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1245890, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6328673, 2.6471305
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4261484, 2.4151304
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8448000, 1.8372324
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4455001, 1.4445721

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376278, upper bound: 1.0385498
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0371912, upper bound: 1.0389872
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9146080, 2.9066935
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2024331, 2.2016947
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8940570, 1.9228520
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3619518, 2.3281574
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1242924
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6482053, 2.6317930
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4168329, 2.4244459
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8353996, 1.8466327
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4445379, 1.4455343

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0391560, upper bound: 1.0370210
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0387186, upper bound: 1.0374576
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9115515, 2.9097500
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2043519, 2.1997757
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8944128, 1.9224961
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3710403, 2.3190689
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6403360, 2.6396618
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4087198
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8434553, 1.8385766
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4491503, 1.4409219

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0413917, upper bound: 1.0347863
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0409543, upper bound: 1.0352229
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9012156, 2.9200861
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1995130, 2.2046151
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9178035, 1.8991053
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3271208, 2.3629880
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6342959, 2.6457019
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4083900, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8345642, 1.8474686
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4446528, 1.4454194

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0385560, upper bound: 1.0376224
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0381194, upper bound: 1.0380597
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8981581, 2.9231429
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2014318, 2.2026963
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9181592, 1.8987494
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3362093, 2.3538997
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1227999, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6264272, 2.6535709
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4241161, 2.4171624
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8426199, 1.8394125
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4492655, 1.4408067

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0407908, upper bound: 1.0353866
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0403542, upper bound: 1.0358240
time: 3.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0358216, upper bound: 1.0403566
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0353842, upper bound: 1.0407933
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0380579, upper bound: 1.0381218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0376199, upper bound: 1.0385584
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0352204, upper bound: 1.0409567
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0347838, upper bound: 1.0413941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0374552, upper bound: 1.0387210
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0370186, upper bound: 1.0391584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0389847, upper bound: 1.0371936
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0385473, upper bound: 1.0376303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0412206, upper bound: 1.0349588
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0407831, upper bound: 1.0353954
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0383835, upper bound: 1.0377936
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0379468, upper bound: 1.0382310
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0406183, upper bound: 1.0355579
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0401817, upper bound: 1.0359953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0359929, upper bound: 1.0401841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0355555, upper bound: 1.0406207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0382286, upper bound: 1.0379493
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0377912, upper bound: 1.0383859
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0353930, upper bound: 1.0407854
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0349564, upper bound: 1.0412228
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0376278, upper bound: 1.0385498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0371912, upper bound: 1.0389872
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0391560, upper bound: 1.0370210
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0387186, upper bound: 1.0374576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0413917, upper bound: 1.0347863
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0409543, upper bound: 1.0352229
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0385560, upper bound: 1.0376224
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0381194, upper bound: 1.0380597
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0407908, upper bound: 1.0353866
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -1.0403542, upper bound: 1.0358240

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9199991, 2.9329123
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2005372, 2.2196276
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8992202, 1.9337649
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3515158, 2.3487906
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6517358, 2.6435156
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4197445, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8651226, 1.8403294
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4419761, 1.4490622

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0358183, upper bound: 1.0393271
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0347857, upper bound: 1.0403532
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9231424, 2.9167655
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2026963, 2.2085774
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9030178, 1.9143622
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3538995, 2.3366160
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1227534
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6535707, 2.6341438
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4165401, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8534143, 1.8426201
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4409382, 1.4492654

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0353809, upper bound: 1.0397566
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0343563, upper bound: 1.0407898
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9169436, 2.9359689
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2024560, 2.2177086
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8995759, 1.9334090
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3606043, 2.3397021
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6438670, 2.6513846
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4229729
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8731782, 1.8322732
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4465888, 1.4444498

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380540, upper bound: 1.0370922
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370205, upper bound: 1.0381185
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9200859, 2.9198222
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2046151, 2.2066586
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9033735, 1.9140062
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3629880, 2.3275278
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6457019, 2.6420128
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4235957
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8614700, 1.8345640
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4455507, 1.4446529

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376166, upper bound: 1.0375218
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0365911, upper bound: 1.0385550
time: 3.73 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0358183, upper bound: 1.0393271
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0347857, upper bound: 1.0403532
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0353809, upper bound: 1.0397566
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0343563, upper bound: 1.0407898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0380540, upper bound: 1.0370922
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0370205, upper bound: 1.0381185
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0376166, upper bound: 1.0375218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.73
Output dim: 9, lower bound: -1.0365911, upper bound: 1.0385550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0352204, upper bound: 1.0409567
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0347838, upper bound: 1.0413941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0374552, upper bound: 1.0387210
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0370186, upper bound: 1.0391584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0389847, upper bound: 1.0371936
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0385473, upper bound: 1.0376303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0412206, upper bound: 1.0349588
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0407831, upper bound: 1.0353954
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0383835, upper bound: 1.0377936
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0379468, upper bound: 1.0382310
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0406183, upper bound: 1.0355579
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0401817, upper bound: 1.0359953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0359929, upper bound: 1.0401841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0355555, upper bound: 1.0406207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0382286, upper bound: 1.0379493
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0377912, upper bound: 1.0383859
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0353930, upper bound: 1.0407854
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0349564, upper bound: 1.0412228
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0376278, upper bound: 1.0385498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0371912, upper bound: 1.0389872
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0391560, upper bound: 1.0370210
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0387186, upper bound: 1.0374576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0413917, upper bound: 1.0347863
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0409543, upper bound: 1.0352229
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0385560, upper bound: 1.0376224
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0381194, upper bound: 1.0380597
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0407908, upper bound: 1.0353866
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.73
Output dim: 9, lower bound: -1.0403542, upper bound: 1.0358240
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=1.4499115943908691
rel_dist={9: [-1.041417153698955, 1.0414178245879198]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262624, upper bound: 0.9264232
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264207, upper bound: 0.9262649
time: 3.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.10
Output dim: 9, lower bound: -0.9262624, upper bound: 0.9264232
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.10
Output dim: 9, lower bound: -0.9264207, upper bound: 0.9262649

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7071352, 2.7259538
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0453281, 2.0533772
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8051741, 1.8014812
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2818260, 2.2842393
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2248368, 2.2211194
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0480428, 2.0503254
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3875437, 2.3958085
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2652578, 2.2784083
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7066634, 1.6925654
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3540778, 1.3537909

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237933, upper bound: 0.9264182
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262574, upper bound: 0.9239468
time: 3.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7095804, 2.7071350
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0463729, 2.0453281
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8014815, 1.8019612
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2821383, 2.2818260
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2211194, 2.2216024
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0483389, 2.0480428
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3886185, 2.3875434
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2669601, 2.2652576
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6925652, 1.6943979
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3537908, 1.3538254

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239443, upper bound: 0.9262598
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264157, upper bound: 0.9237958
time: 3.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 9, lower bound: -0.9237933, upper bound: 0.9264182
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 9, lower bound: -0.9262574, upper bound: 0.9239468
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 9, lower bound: -0.9239443, upper bound: 0.9262598
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 9, lower bound: -0.9264157, upper bound: 0.9237958

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6981249, 2.7084031
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0428400, 2.0498726
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8149168, 1.8079476
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2617078, 2.2701411
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2246652, 2.2211823
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0531778, 2.0540690
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3861887, 2.3894448
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2613721, 2.2729423
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7025592, 1.6867659
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3461342, 1.3487759

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237908, upper bound: 0.9259484
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233241, upper bound: 0.9264155
time: 3.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6895843, 2.7169449
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0418234, 2.0508897
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8116400, 1.8112245
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2677279, 2.2641213
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2248998, 2.2209477
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0517859, 2.0554605
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3811800, 2.3944507
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2597919, 2.2745240
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7008636, 1.6884615
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3490627, 1.3458474

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262550, upper bound: 0.9234768
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9257880, upper bound: 0.9239442
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7005706, 2.6895843
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0438843, 2.0418234
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8112242, 1.8084273
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2620201, 2.2677281
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2209477, 2.2216654
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0534744, 2.0517862
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3872640, 2.3811798
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2630754, 2.2597916
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6884620, 1.6885986
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3458474, 1.3488102

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239416, upper bound: 0.9257905
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234743, upper bound: 0.9262575
time: 3.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6920300, 2.6981246
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0428677, 2.0428400
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8079474, 1.8117042
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2680407, 2.2617080
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2211823, 2.2214308
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0520830, 2.0531778
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3822548, 2.3861890
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2614951, 2.2613721
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6867664, 1.6902943
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3487756, 1.3458817

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264130, upper bound: 0.9233247
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259459, upper bound: 0.9237917
time: 4.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9237908, upper bound: 0.9259484
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9233241, upper bound: 0.9264155
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9262550, upper bound: 0.9234768
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9257880, upper bound: 0.9239442
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9239416, upper bound: 0.9257905
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9234743, upper bound: 0.9262575
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9264130, upper bound: 0.9233247
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.97
Output dim: 9, lower bound: -0.9259459, upper bound: 0.9237917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6727047, 2.6725667
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0372934, 2.0420544
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7540786, 1.7655790
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2040873, 2.1854298
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2340894, 2.2380457
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0293727, 2.0204720
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3981051, 2.3905425
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2828031, 2.2878063
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7009525, 1.6845100
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3463397, 1.3490708

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221724, upper bound: 0.9259454
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237898, upper bound: 0.9243400
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6622882, 2.6829832
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0350218, 2.0443258
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7725484, 1.7471092
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1769967, 2.2125204
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2415290, 2.2306068
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0195808, 2.0302639
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3872867, 2.4013622
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2762361, 2.2943728
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7003026, 1.6851599
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3464291, 1.3489814

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217171, upper bound: 0.9264127
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233231, upper bound: 0.9247982
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6641645, 2.6811085
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0362768, 2.0430713
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7508013, 1.7688558
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2101073, 2.1794100
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2343240, 2.2378111
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0279813, 2.0218635
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3930955, 2.3955483
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2812223, 2.2893879
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6992579, 1.6862054
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3492682, 1.3461423

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246362, upper bound: 0.9234739
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262540, upper bound: 0.9218704
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6537480, 2.6915252
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0340052, 2.0453429
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7692711, 1.7503860
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1830168, 2.2065005
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2417636, 2.2303722
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0181894, 2.0316553
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3822775, 2.4063680
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2746558, 2.2959545
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6986074, 1.6868556
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3493576, 1.3460529

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241810, upper bound: 0.9239414
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9257870, upper bound: 0.9223254
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6751504, 2.6537480
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0383372, 2.0340052
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7503860, 1.7660587
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2043991, 2.1830168
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2303720, 2.2385297
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0296698, 2.0181892
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3991795, 2.3822775
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2845054, 2.2746556
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6868553, 1.6863427
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3460529, 1.3491049

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223246, upper bound: 0.9257876
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239406, upper bound: 0.9241812
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6647339, 2.6641645
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0360661, 2.0362766
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7688558, 1.7475889
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1773086, 2.2101073
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2378106, 2.2310908
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0198784, 2.0279810
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3883610, 2.3930955
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2779388, 2.2812221
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6862054, 1.6869926
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3461423, 1.3490152

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9218679, upper bound: 0.9262564
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234733, upper bound: 0.9246387
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6666102, 2.6622884
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0373211, 2.0350215
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7471092, 1.7693355
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2104192, 2.1769967
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2306066, 2.2382951
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0282784, 2.0195808
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3941698, 2.3872867
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2829247, 2.2762361
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6851597, 1.6880383
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3489811, 1.3461764

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9247957, upper bound: 0.9233236
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264120, upper bound: 0.9217173
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6561937, 2.6727049
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0350494, 2.0372932
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7655790, 1.7508657
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1833286, 2.2040873
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2380452, 2.2308562
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0184865, 2.0293727
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3833518, 2.3981049
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2763581, 2.2828026
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6845098, 1.6886883
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3490708, 1.3460870

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243395, upper bound: 0.9237924
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259449, upper bound: 0.9221750
time: 4.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9221724, upper bound: 0.9259454
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9237898, upper bound: 0.9243400
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9217171, upper bound: 0.9264127
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9233231, upper bound: 0.9247982
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9246362, upper bound: 0.9234739
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9262540, upper bound: 0.9218704
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9241810, upper bound: 0.9239414
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9257870, upper bound: 0.9223254
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9223246, upper bound: 0.9257876
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9239406, upper bound: 0.9241812
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9218679, upper bound: 0.9262564
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9234733, upper bound: 0.9246387
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9247957, upper bound: 0.9233236
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9264120, upper bound: 0.9217173
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9243395, upper bound: 0.9237924
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.74
Output dim: 9, lower bound: -0.9259449, upper bound: 0.9221750

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6681423, 2.6656270
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0410771, 2.0473313
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7547815, 1.7665582
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1798320, 2.1682434
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2188864, 2.2165840
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0277181, 2.0108974
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3831906, 2.3695080
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2408628, 2.2580974
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6852646, 1.6750875
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3449738, 1.3512923

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221640, upper bound: 0.9254848
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217151, upper bound: 0.9259371
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6657653, 2.6680045
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0425706, 2.0458388
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7550581, 1.7662811
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1869006, 2.1611745
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2126274, 2.2228429
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0197978, 2.0188174
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3770704, 2.3756285
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2530937, 2.2458661
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6915302, 1.6688216
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3485610, 1.3477048

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237814, upper bound: 0.9238785
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233280, upper bound: 0.9243314
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6577263, 2.6760435
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0388064, 2.0496030
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7732513, 1.7480884
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1527414, 2.1953340
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2263260, 2.2091453
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0179257, 2.0206892
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3723722, 2.3803277
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2342958, 2.2646639
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6846142, 1.6757376
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3450630, 1.3512028

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217087, upper bound: 0.9259527
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9212561, upper bound: 0.9264060
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6553488, 2.6784210
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0402989, 2.0481102
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7735279, 1.7478116
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1598105, 2.1882651
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2200670, 2.2154040
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0100060, 2.0286093
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3662524, 2.3864481
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2465277, 2.2524326
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6908798, 1.6694715
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3486507, 1.3476151

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233146, upper bound: 0.9243408
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9228605, upper bound: 0.9247897
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6596022, 2.6741688
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0400615, 2.0483484
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7515037, 1.7698350
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1858521, 2.1622233
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2191210, 2.2163494
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0263262, 2.0122888
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3781815, 2.3745141
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2392817, 2.2596791
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6835690, 1.6767828
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3479021, 1.3483638

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246278, upper bound: 0.9230132
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241789, upper bound: 0.9234673
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6572247, 2.6765463
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0415540, 2.0468559
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7517812, 1.7695582
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1929207, 2.1551547
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2128620, 2.2226083
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0184064, 2.0202088
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3720613, 2.3806343
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2515135, 2.2474477
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6898355, 1.6705172
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3514898, 1.3447763

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262455, upper bound: 0.9214089
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9257921, upper bound: 0.9218620
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6491857, 2.6845856
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0377898, 2.0506201
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7699735, 1.7513652
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1587615, 2.1893139
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2265606, 2.2089107
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0165343, 2.0220807
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3673635, 2.3853338
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2327156, 2.2662456
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6829195, 1.6774330
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3479917, 1.3482744

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241726, upper bound: 0.9234813
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237202, upper bound: 0.9239347
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6468081, 2.6869628
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0392823, 2.0491273
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7702510, 1.7510884
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1658301, 2.1822453
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2203016, 2.2151694
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0086145, 2.0300007
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3612428, 2.3914540
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2449474, 2.2540143
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6891851, 1.6711671
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3515790, 1.3446869

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9257786, upper bound: 0.9218680
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9253244, upper bound: 0.9223164
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6705880, 2.6468081
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0421214, 2.0392823
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7510889, 1.7670379
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1801443, 2.1658301
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2151690, 2.2170672
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0280142, 2.0086145
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3842654, 2.3612430
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2425652, 2.2449470
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6711674, 1.6769202
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3446867, 1.3513263

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223162, upper bound: 0.9253268
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9218673, upper bound: 0.9257793
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6682105, 2.6491857
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0436149, 2.0377896
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7513654, 1.7667608
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1872129, 2.1587615
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2089100, 2.2233262
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0200949, 2.0165346
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3781452, 2.3673635
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2547970, 2.2327154
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6774330, 1.6706543
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3482745, 1.3477389

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239322, upper bound: 0.9237200
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234788, upper bound: 0.9241731
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6601715, 2.6572249
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0398507, 2.0415537
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7695587, 1.7485681
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1530533, 2.1929207
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2226076, 2.2096283
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0182228, 2.0184064
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3734469, 2.3720613
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2359982, 2.2515132
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6705170, 1.6775701
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3447764, 1.3512369

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9218595, upper bound: 0.9257946
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9214064, upper bound: 0.9262480
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6577945, 2.6596022
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0413432, 2.0400612
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7698352, 1.7482913
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1601219, 2.1858521
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2163496, 2.2158873
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0103025, 2.0263264
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3673272, 2.3781815
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2482300, 2.2392819
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6767826, 1.6713042
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3483636, 1.3476495

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234648, upper bound: 0.9241813
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9230107, upper bound: 0.9246303
time: 4.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6620479, 2.6553485
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0411057, 2.0402987
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7478120, 1.7703147
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1861639, 2.1598103
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2154036, 2.2168326
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0266228, 2.0100060
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3792562, 2.3662524
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2409849, 2.2465274
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6694717, 1.6786158
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3476152, 1.3483979

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9247873, upper bound: 0.9228613
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243384, upper bound: 0.9233171
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6596704, 2.6577260
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0425982, 2.0388062
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7480886, 1.7700379
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1932325, 2.1527414
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2091446, 2.2230916
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0187035, 2.0179260
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3731360, 2.3723726
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2532158, 2.2342958
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6757374, 1.6723497
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3512027, 1.3448104

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264035, upper bound: 0.9212561
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259501, upper bound: 0.9217093
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6516314, 2.6657653
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0388341, 2.0425704
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7662818, 1.7518449
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1590738, 2.1869006
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2228432, 2.2093937
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0168314, 2.0197980
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3684382, 2.3770704
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2344179, 2.2530940
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6688213, 1.6792657
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3477046, 1.3483084

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243311, upper bound: 0.9233303
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238780, upper bound: 0.9237839
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6492538, 2.6681426
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0403266, 2.0410776
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7665584, 1.7515681
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1661425, 2.1798320
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2165842, 2.2156527
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0089111, 2.0277181
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3623176, 2.3831909
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2466497, 2.2408624
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6750870, 1.6729999
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3512924, 1.3447210

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259364, upper bound: 0.9217157
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9254823, upper bound: 0.9221648
time: 3.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9221640, upper bound: 0.9254848
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9217151, upper bound: 0.9259371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9237814, upper bound: 0.9238785
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9233280, upper bound: 0.9243314
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9217087, upper bound: 0.9259527
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9212561, upper bound: 0.9264060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9233146, upper bound: 0.9243408
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9228605, upper bound: 0.9247897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9246278, upper bound: 0.9230132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9241789, upper bound: 0.9234673
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9262455, upper bound: 0.9214089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9257921, upper bound: 0.9218620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9241726, upper bound: 0.9234813
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9237202, upper bound: 0.9239347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9257786, upper bound: 0.9218680
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9253244, upper bound: 0.9223164
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9223162, upper bound: 0.9253268
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9218673, upper bound: 0.9257793
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9239322, upper bound: 0.9237200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9234788, upper bound: 0.9241731
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9218595, upper bound: 0.9257946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9214064, upper bound: 0.9262480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9234648, upper bound: 0.9241813
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9230107, upper bound: 0.9246303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9247873, upper bound: 0.9228613
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9243384, upper bound: 0.9233171
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9264035, upper bound: 0.9212561
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9259501, upper bound: 0.9217093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9243311, upper bound: 0.9233303
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9238780, upper bound: 0.9237839
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9259364, upper bound: 0.9217157
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.41
Output dim: 9, lower bound: -0.9254823, upper bound: 0.9221648

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6649995, 2.6750426
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0389190, 2.0537667
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7509840, 1.7778518
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1774483, 2.1753285
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2184100, 2.2179894
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0250330, 2.0189373
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3813558, 2.3749623
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2427320, 2.2574749
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6920803, 1.6727967
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3455775, 1.3510889

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221614, upper bound: 0.9246861
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213681, upper bound: 0.9254822
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6681423, 2.6624842
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0410771, 2.0451725
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7547815, 1.7627609
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1798320, 2.1658595
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2188864, 2.2161074
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0277181, 2.0082126
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3831906, 2.3676732
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2402401, 2.2580974
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6829736, 1.6750875
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3447702, 1.3512923

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217125, upper bound: 0.9251424
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9209190, upper bound: 0.9259364
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6626220, 2.6774201
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0404115, 2.0522742
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7512605, 1.7775750
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1845169, 2.1682596
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2121511, 2.2242482
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0171127, 2.0268574
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3752356, 2.3810825
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2549639, 2.2452433
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6983459, 1.6665308
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3491652, 1.3475016

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237788, upper bound: 0.9230814
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9229880, upper bound: 0.9238761
time: 4.04 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9221614, upper bound: 0.9246861
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9213681, upper bound: 0.9254822
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9217125, upper bound: 0.9251424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9209190, upper bound: 0.9259364
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9237788, upper bound: 0.9230814
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -0.9229880, upper bound: 0.9238761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9233280, upper bound: 0.9243314
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9217087, upper bound: 0.9259527
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9212561, upper bound: 0.9264060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9233146, upper bound: 0.9243408
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9228605, upper bound: 0.9247897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9246278, upper bound: 0.9230132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9241789, upper bound: 0.9234673
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9262455, upper bound: 0.9214089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9257921, upper bound: 0.9218620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9241726, upper bound: 0.9234813
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9237202, upper bound: 0.9239347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9257786, upper bound: 0.9218680
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9253244, upper bound: 0.9223164
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9223162, upper bound: 0.9253268
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9218673, upper bound: 0.9257793
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9239322, upper bound: 0.9237200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9234788, upper bound: 0.9241731
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9218595, upper bound: 0.9257946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9214064, upper bound: 0.9262480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9234648, upper bound: 0.9241813
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9230107, upper bound: 0.9246303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9247873, upper bound: 0.9228613
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9243384, upper bound: 0.9233171
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9264035, upper bound: 0.9212561
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9259501, upper bound: 0.9217093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9243311, upper bound: 0.9233303
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9238780, upper bound: 0.9237839
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9259364, upper bound: 0.9217157
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -0.9254823, upper bound: 0.9221648
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=1.353825569152832
rel_dist={9: [-0.9264248152800034, 0.9264272454017544]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658477, upper bound: 0.8660059
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660033, upper bound: 0.8658487
time: 3.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.13
Output dim: 9, lower bound: -0.8658477, upper bound: 0.8660059
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.13
Output dim: 9, lower bound: -0.8660033, upper bound: 0.8658487

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5826826, 2.5988131
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9649882, 1.9718878
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7315238, 1.7283587
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1986623, 2.2007308
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1604314, 2.1572456
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9910984, 1.9930551
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2554893, 2.2625732
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1782713, 2.1895435
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6217787, 1.6096945
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3059936, 1.3057479

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8637255, upper bound: 0.8660000
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658434, upper bound: 0.8638821
time: 3.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5851278, 2.5826826
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9660335, 1.9649885
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7283595, 1.7288387
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1989751, 2.1986623
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1572452, 2.1577287
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9913950, 1.9910984
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2565641, 2.2554891
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1799741, 2.1782715
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6096947, 1.6115267
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3057480, 1.3057824

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8638811, upper bound: 0.8658443
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659990, upper bound: 0.8637264
time: 3.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 9, lower bound: -0.8637255, upper bound: 0.8660000
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 9, lower bound: -0.8658434, upper bound: 0.8638821
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 9, lower bound: -0.8638811, upper bound: 0.8658443
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 9, lower bound: -0.8659990, upper bound: 0.8637264

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5724521, 2.5812624
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9623547, 1.9683831
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7407987, 1.7348249
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1785440, 2.1857727
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1602597, 2.1572747
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9960346, 1.9967985
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2534187, 2.2562094
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1741605, 2.1840775
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6174314, 1.6038949
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2980502, 1.3003144

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8637229, upper bound: 0.8655669
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8632923, upper bound: 0.8659972
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5651321, 2.5885839
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9614840, 1.9692550
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7379897, 1.7376337
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1837039, 2.1806128
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1604609, 2.1570735
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9948416, 1.9979911
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2491252, 2.2605002
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1728053, 2.1854331
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6159790, 1.6053481
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3005602, 1.2978044

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658408, upper bound: 0.8634490
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654102, upper bound: 0.8638810
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5748978, 2.5651321
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9633999, 1.9614840
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7376339, 1.7353048
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1788568, 2.1837044
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1570735, 2.1577578
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9963312, 1.9948421
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2544940, 2.2491252
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1758637, 2.1728055
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6053483, 1.6057274
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2978044, 1.3003488

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8638784, upper bound: 0.8654128
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8634464, upper bound: 0.8658434
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5675778, 2.5724523
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9625282, 1.9623551
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7348254, 1.7381134
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1840162, 2.1785443
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1572747, 2.1575565
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9951386, 1.9960346
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2502000, 2.2534187
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1745095, 2.1741602
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6038949, 1.6071808
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3003144, 1.2978387

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659963, upper bound: 0.8632949
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655643, upper bound: 0.8637237
time: 4.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8637229, upper bound: 0.8655669
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8632923, upper bound: 0.8659972
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8658408, upper bound: 0.8634490
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8654102, upper bound: 0.8638810
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8638784, upper bound: 0.8654128
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8634464, upper bound: 0.8658434
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8659963, upper bound: 0.8632949
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 9, lower bound: -0.8655643, upper bound: 0.8637237

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5455441, 2.5454259
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9564838, 1.9605649
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6799605, 1.6898177
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1170535, 2.1010613
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1696844, 2.1730754
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9708309, 1.9632015
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2637892, 2.2573071
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1946530, 2.1989415
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6157327, 1.6016388
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2982557, 1.3005967

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8618899, upper bound: 0.8655636
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8637219, upper bound: 0.8637652
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5366158, 2.5543544
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9545364, 1.9625118
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6957915, 1.6739867
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0938330, 2.1242819
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1760612, 2.1666992
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9624376, 1.9715946
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2545166, 2.2665811
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1890244, 2.2045698
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6151752, 1.6021960
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2983325, 1.3005199

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8614891, upper bound: 0.8659981
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8632914, upper bound: 0.8641648
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5382242, 2.5527477
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9556122, 1.9614365
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6771514, 1.6926265
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1222134, 2.0959015
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1698852, 2.1728742
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9696383, 1.9643941
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2594957, 2.2615979
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1932979, 2.2002971
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6142797, 1.6030922
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3007658, 1.2980866

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8640078, upper bound: 0.8634458
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658398, upper bound: 0.8616473
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5292954, 2.5616760
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9536657, 1.9633837
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6929829, 1.6767952
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0989928, 2.1191220
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1762619, 2.1664982
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9612451, 1.9727874
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2502227, 2.2708721
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1876693, 2.2059255
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6137228, 1.6036493
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3008425, 1.2980099

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8636070, upper bound: 0.8638801
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654093, upper bound: 0.8620484
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5479898, 2.5292957
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9575286, 1.9536655
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6767952, 1.6902974
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1173654, 2.0989928
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1664982, 2.1735594
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9711280, 1.9612451
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2648635, 2.2502227
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1963553, 2.1876695
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6036491, 1.6034715
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2980099, 1.3006308

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8620457, upper bound: 0.8654109
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8638775, upper bound: 0.8636097
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5390615, 2.5382242
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9555812, 1.9556127
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6926262, 1.6744664
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0941448, 2.1222134
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1728740, 2.1671832
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9627352, 1.9696379
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2555909, 2.2594957
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1907268, 2.1932979
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6030922, 1.6040287
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2980864, 1.3005540

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8616447, upper bound: 0.8658410
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8634455, upper bound: 0.8640088
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5406699, 2.5366158
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9566569, 1.9545369
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6739867, 1.6931062
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1225252, 2.0938330
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1666989, 2.1733582
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9699354, 1.9624376
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2605700, 2.2545164
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1950006, 2.1890242
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6021957, 1.6049249
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3005199, 1.2981205

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8641636, upper bound: 0.8632940
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659954, upper bound: 0.8614917
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5317411, 2.5455444
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9547100, 1.9564838
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6898177, 1.6772749
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0993047, 2.1170535
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1730752, 2.1669822
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9615421, 1.9708307
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2512970, 2.2637892
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1893721, 2.1946526
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6016388, 1.6054820
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3005967, 1.2980440

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8637626, upper bound: 0.8637231
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655634, upper bound: 0.8618908
time: 3.94 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8618899, upper bound: 0.8655636
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8637219, upper bound: 0.8637652
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8614891, upper bound: 0.8659981
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8632914, upper bound: 0.8641648
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8640078, upper bound: 0.8634458
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8658398, upper bound: 0.8616473
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8636070, upper bound: 0.8638801
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8654093, upper bound: 0.8620484
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8620457, upper bound: 0.8654109
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8638775, upper bound: 0.8636097
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8616447, upper bound: 0.8658410
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8634455, upper bound: 0.8640088
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8641636, upper bound: 0.8632940
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8659954, upper bound: 0.8614917
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8637626, upper bound: 0.8637231
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.17
Output dim: 9, lower bound: -0.8655634, upper bound: 0.8618908

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5406427, 2.5384862
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9602685, 1.9656286
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6806629, 1.6907573
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0927982, 2.0828650
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1535873, 2.1516137
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9680448, 1.9536269
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2480006, 2.2362728
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1527128, 2.1674852
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6000443, 1.5913212
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2968895, 1.3023056

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8618826, upper bound: 0.8650754
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8613967, upper bound: 0.8655586
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5317140, 2.5474148
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9583211, 1.9675758
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6964939, 1.6749263
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0695777, 2.1060855
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1599646, 2.1452377
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9596515, 1.9620199
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2387280, 2.2455468
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1470842, 2.1731136
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5994873, 1.5918784
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2969663, 1.3022288

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8614818, upper bound: 0.8655056
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609968, upper bound: 0.8659883
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5312843, 2.5478458
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9606767, 1.9652212
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6780918, 1.6933289
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1040173, 2.0716462
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1484232, 2.1567774
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9600635, 1.9616079
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2384610, 2.2458096
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1618423, 2.1583569
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6039619, 1.5874038
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3024747, 1.2967204

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658326, upper bound: 0.8611550
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8653475, upper bound: 0.8616382
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5223560, 2.5567741
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9587293, 1.9671681
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6939228, 1.6774976
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0807967, 2.0948668
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1548004, 2.1504014
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9516702, 1.9700012
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2291884, 2.2550836
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1562138, 2.1639855
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6034050, 1.5879608
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3025515, 1.2966439

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654020, upper bound: 0.8615552
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8649206, upper bound: 0.8620396
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5430880, 2.5223560
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9613128, 1.9587295
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6774976, 1.6912370
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0931101, 2.0807967
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1504011, 2.1520970
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9683409, 1.9516702
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2490754, 2.2291884
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1544151, 2.1562133
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5879612, 1.5931540
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2966440, 1.3023399

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8620385, upper bound: 0.8649214
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8615525, upper bound: 0.8654044
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5341597, 2.5312843
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9593654, 1.9606764
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6933296, 1.6754060
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0698900, 2.1040173
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1567774, 2.1457210
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9599485, 1.9600632
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2398028, 2.2384613
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1487865, 2.1618419
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5874033, 1.5937111
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2967205, 1.3022631

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8616374, upper bound: 0.8653501
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8611524, upper bound: 0.8658329
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5337300, 2.5317140
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9617209, 1.9583213
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6749265, 1.6938086
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1043291, 2.0695777
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1452370, 2.1572607
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9603596, 1.9596515
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2395358, 2.2387280
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1635447, 2.1470840
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5918779, 1.5892365
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3022289, 1.2967547

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659881, upper bound: 0.8609994
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655031, upper bound: 0.8614844
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5248017, 2.5406425
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9597735, 1.9602685
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6907575, 1.6779773
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0811090, 2.0927982
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1516132, 2.1508846
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9519672, 1.9680445
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2302632, 2.2480009
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1579161, 2.1527123
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5913210, 1.5897936
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3023057, 1.2966779

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655561, upper bound: 0.8613993
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8650747, upper bound: 0.8618835
time: 4.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8618826, upper bound: 0.8650754
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8613967, upper bound: 0.8655586
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8614818, upper bound: 0.8655056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8609968, upper bound: 0.8659883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8658326, upper bound: 0.8611550
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8653475, upper bound: 0.8616382
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8654020, upper bound: 0.8615552
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8649206, upper bound: 0.8620396
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8620385, upper bound: 0.8649214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8615525, upper bound: 0.8654044
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8616374, upper bound: 0.8653501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8611524, upper bound: 0.8658329
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8659881, upper bound: 0.8609994
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8655031, upper bound: 0.8614844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8655561, upper bound: 0.8613993
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.37
Output dim: 9, lower bound: -0.8650747, upper bound: 0.8618835

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5406427, 2.5353434
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9602685, 1.9634697
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6806629, 1.6869602
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0927982, 2.0804811
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1535873, 2.1511371
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9680448, 1.9509420
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2480006, 2.2344377
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1520901, 2.1674852
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5977533, 1.5913212
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2966864, 1.3023056

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8613944, upper bound: 0.8648700
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8607069, upper bound: 0.8655564
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5285711, 2.5550363
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9561620, 1.9727833
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6926973, 1.6840641
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0671940, 2.1118178
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1594877, 2.1463742
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9569664, 1.9685278
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2368932, 2.2499597
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1485977, 2.1724911
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6050022, 1.5895877
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2974551, 1.3020257

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8614796, upper bound: 0.8648167
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8607927, upper bound: 0.8655017
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5317140, 2.5442719
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9583211, 1.9654167
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6964939, 1.6711290
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0695777, 2.1037016
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1599646, 2.1447608
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9596515, 1.9593351
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2387280, 2.2437117
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1464615, 2.1731136
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5971963, 1.5918784
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2967632, 1.3022288

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609945, upper bound: 0.8653002
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8603069, upper bound: 0.8659868
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5281415, 2.5554674
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9585176, 1.9704287
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6742942, 1.7024670
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1016331, 2.0773785
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1479468, 2.1579139
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9573784, 1.9681160
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2366261, 2.2502224
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1633558, 2.1577342
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6094768, 1.5851130
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3029635, 1.2965173

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658303, upper bound: 0.8604654
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8651435, upper bound: 0.8611527
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5312843, 2.5447030
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9606767, 1.9630620
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6780918, 1.6895318
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1040173, 2.0692623
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1484232, 2.1563008
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9600635, 1.9589233
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2384610, 2.2439744
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1612196, 2.1583569
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6016719, 1.5874038
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3022716, 1.2967204

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8653453, upper bound: 0.8609509
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8646585, upper bound: 0.8616377
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5192127, 2.5643959
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9565701, 1.9723756
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6901252, 1.6866357
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0784130, 2.1005991
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1543241, 2.1515379
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9489851, 1.9765091
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2273536, 2.2594965
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1577272, 2.1633627
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6089199, 1.5856701
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3030403, 1.2964405

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8653998, upper bound: 0.8608654
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8647155, upper bound: 0.8615512
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5430880, 2.5192132
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9613128, 1.9565704
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6774976, 1.6874399
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0931101, 2.0784128
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1504011, 2.1516204
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9683409, 1.9489856
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2490754, 2.2273533
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1537924, 2.1562133
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5856702, 1.5931540
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2964406, 1.3023399

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8615503, upper bound: 0.8647180
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8608628, upper bound: 0.8654006
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5310168, 2.5389061
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9572072, 1.9658840
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6895320, 1.6845441
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0675058, 2.1097496
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1563010, 2.1468573
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9572635, 1.9665713
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2379680, 2.2428741
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1503000, 2.1612194
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5929182, 1.5914204
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2972090, 1.3020598

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8616352, upper bound: 0.8646611
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609483, upper bound: 0.8653479
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5341597, 2.5281415
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9593654, 1.9585173
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6933296, 1.6716089
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0698900, 2.1016333
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1567774, 2.1452441
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9599485, 1.9573784
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2398028, 2.2366264
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1481638, 2.1618419
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5851133, 1.5937111
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2965171, 1.3022631

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8611501, upper bound: 0.8651460
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8604628, upper bound: 0.8658313
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5305872, 2.5393357
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9595618, 1.9635291
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6711290, 1.7029467
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1019454, 2.0753100
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1447606, 2.1583972
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9576750, 1.9661593
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2377009, 2.2431409
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1650581, 2.1464612
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5973928, 1.5869458
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3027174, 1.2965513

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659859, upper bound: 0.8603095
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8652990, upper bound: 0.8609971
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5337300, 2.5285711
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9617209, 1.9561622
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6749265, 1.6900115
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1043291, 2.0671937
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1452370, 2.1567841
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9603596, 1.9569666
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2395358, 2.2368932
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1629219, 2.1470840
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5895879, 1.5892365
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3020256, 1.2967547

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655008, upper bound: 0.8607952
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648141, upper bound: 0.8614821
time: 4.20 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8613944, upper bound: 0.8648700
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8607069, upper bound: 0.8655564
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8614796, upper bound: 0.8648167
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8607927, upper bound: 0.8655017
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8609945, upper bound: 0.8653002
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8603069, upper bound: 0.8659868
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8658303, upper bound: 0.8604654
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8651435, upper bound: 0.8611527
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8653453, upper bound: 0.8609509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8646585, upper bound: 0.8616377
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8653998, upper bound: 0.8608654
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8647155, upper bound: 0.8615512
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8615503, upper bound: 0.8647180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8608628, upper bound: 0.8654006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8616352, upper bound: 0.8646611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8609483, upper bound: 0.8653479
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8611501, upper bound: 0.8651460
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8604628, upper bound: 0.8658313
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8659859, upper bound: 0.8603095
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8652990, upper bound: 0.8609971
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8655008, upper bound: 0.8607952
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.67
Output dim: 9, lower bound: -0.8648141, upper bound: 0.8614821
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.67
Output dim: 9, lower bound: -0.8655561, upper bound: 0.8613993
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2414.24 seconds
