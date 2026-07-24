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
execution time: IAR + LP analysis = 13.63 + 36.67 = 50.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.2026377, upper bound: 1.2026362


# Binary Search by BASE starts (time budget: 3549.70 seconds, max iter: 100)

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
Binary search time: 207.47 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 3342.23 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382476, upper bound: 1.0414132
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0382500
time: 3.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 9, lower bound: -1.0382476, upper bound: 1.0414132
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 9, lower bound: -1.0414107, upper bound: 1.0382500

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9519153, 2.9409347
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2048540, 2.2035472
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9588852, 1.9546723
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4283476, 2.4360881
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6528046, 2.6463642
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8565214, 1.8543410
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4419675, 1.4457328

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382442, upper bound: 1.0403778
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0372123, upper bound: 1.0414098
time: 3.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9409347, 2.9519153
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2035475, 2.2048542
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9546723, 1.9588854
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4360876, 2.4283481
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6463645, 2.6528046
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8543413, 1.8565211
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4457328, 1.4419675

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414073, upper bound: 1.0372147
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0403754, upper bound: 1.0382466
time: 3.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 9, lower bound: -1.0382442, upper bound: 1.0403778
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 9, lower bound: -1.0372123, upper bound: 1.0414098
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 9, lower bound: -1.0414073, upper bound: 1.0372147
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.29
Output dim: 9, lower bound: -1.0403754, upper bound: 1.0382466

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9388700, 2.9384568
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2017536, 2.2029550
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9523129, 1.9534345
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4211082, 2.4347193
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6442215, 2.6447389
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8558974, 1.8510752
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4415703, 1.4436169

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380682, upper bound: 1.0403727
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382392, upper bound: 1.0402002
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9494376, 2.9278898
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2042618, 2.2004464
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9576478, 1.9481001
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4269791, 2.4288485
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6511796, 2.6377809
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8532548, 1.8537176
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4398518, 1.4453354

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0349858, upper bound: 1.0414082
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0372107, upper bound: 1.0391823
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9278903, 2.9494371
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2004461, 2.2042620
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9481001, 1.9576478
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4288483, 2.4269793
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6377809, 2.6511793
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8537173, 1.8532550
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4453354, 1.4398518

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412308, upper bound: 1.0372096
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414022, upper bound: 1.0370372
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9384570, 2.9388702
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2029552, 2.2017534
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9534345, 1.9523129
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4347191, 2.4211085
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6447389, 2.6442211
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8510752, 1.8558977
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4436169, 1.4415703

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0403648, upper bound: 1.0377988
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0399353, upper bound: 1.0382361
time: 3.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0380682, upper bound: 1.0403727
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0382392, upper bound: 1.0402002
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0349858, upper bound: 1.0414082
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0372107, upper bound: 1.0391823
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0412308, upper bound: 1.0372096
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0414022, upper bound: 1.0370372
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0403648, upper bound: 1.0377988
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.82
Output dim: 9, lower bound: -1.0399353, upper bound: 1.0382361

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9364257, 2.9602075
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2007089, 2.2122593
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9565809, 1.9529550
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4207959, 2.4375093
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6431472, 2.6542909
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8721900, 1.8492424
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4419050, 1.4435828

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380577, upper bound: 1.0399326
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376201, upper bound: 1.0403621
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9388700, 2.9360123
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2017536, 2.2019103
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9518330, 1.9534345
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4211082, 2.4344068
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6442215, 2.6436646
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8540649, 1.8510752
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4415362, 1.4436169

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382368, upper bound: 1.0395958
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376360, upper bound: 1.0401977
time: 3.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9469347, 2.9239397
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2047472, 2.2012784
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9581652, 1.9489853
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4199033, 2.4244244
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6461787, 2.6298382
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4244633, 2.4275088
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8464167, 1.8494463
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4366286, 1.4433156

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0348082, upper bound: 1.0414031
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0349808, upper bound: 1.0412317
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9454870, 2.9253869
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2050943, 2.2009315
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9585333, 1.9486170
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4225550, 2.4217730
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6432366, 2.6327801
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4227667
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8489830, 1.8468795
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4378316, 1.4421123

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370331, upper bound: 1.0391771
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0372057, upper bound: 1.0390059
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9254451, 2.9711900
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1994014, 2.2135668
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9523675, 1.9571679
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4285359, 2.4297695
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6367066, 2.6607273
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8700109, 1.8514223
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4456701, 1.4398174

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0390035, upper bound: 1.0372080
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412292, upper bound: 1.0349832
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9278903, 2.9469924
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2004461, 2.2032175
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9476202, 1.9576478
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4288483, 2.4266667
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6377809, 2.6501052
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8518848, 1.8532550
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4453015, 1.4398518

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0413999, upper bound: 1.0364328
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0407990, upper bound: 1.0370348
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9353118, 2.9518719
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2007952, 2.2106435
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9496374, 1.9626331
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4323349, 2.4308987
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6429036, 2.6517577
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8604922, 1.8536067
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4444513, 1.4413669

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0403624, upper bound: 1.0371964
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0397604, upper bound: 1.0377966
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9384570, 2.9357247
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2029552, 2.1995935
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9534345, 1.9485157
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4347191, 2.4187241
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6447389, 2.6423860
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8487840, 1.8558977
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4434135, 1.4415703

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376994, upper bound: 1.0382351
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0399342, upper bound: 1.0359994
time: 3.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0380577, upper bound: 1.0399326
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0376201, upper bound: 1.0403621
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0382368, upper bound: 1.0395958
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0376360, upper bound: 1.0401977
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0348082, upper bound: 1.0414031
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0349808, upper bound: 1.0412317
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0370331, upper bound: 1.0391771
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0372057, upper bound: 1.0390059
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0390035, upper bound: 1.0372080
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0412292, upper bound: 1.0349832
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0413999, upper bound: 1.0364328
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0407990, upper bound: 1.0370348
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0403624, upper bound: 1.0371964
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0397604, upper bound: 1.0377966
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0376994, upper bound: 1.0382351
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.60
Output dim: 9, lower bound: -1.0399342, upper bound: 1.0359994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9332786, 2.9732046
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1985493, 2.2211499
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9527838, 1.9626331
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4184117, 2.4472992
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6413109, 2.6618273
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8816080, 1.8469512
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4427395, 1.4433794

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380552, upper bound: 1.0393282
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0374530, upper bound: 1.0399302
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9364257, 2.9570611
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2007089, 2.2100997
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9565809, 1.9491575
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4207959, 2.4351251
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6431472, 2.6524553
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8698988, 1.8492424
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4417017, 1.4435828

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376177, upper bound: 1.0397577
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370164, upper bound: 1.0403596
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9164271, 2.9001751
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1968555, 2.1940918
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8909953, 1.9163430
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3712273, 2.3496957
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6592274, 2.6447616
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8526447, 1.8488190
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4417415, 1.4439373

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0360000, upper bound: 1.0395947
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0382357, upper bound: 1.0373599
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9030337, 2.9135680
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1939354, 2.1970122
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9147418, 1.8925962
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3363972, 2.3845263
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6453185, 2.6586709
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8518093, 1.8496547
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4418566, 1.4438224

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0354000, upper bound: 1.0401967
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376348, upper bound: 1.0379620
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9444895, 2.9456899
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2037029, 2.2105832
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9624321, 1.9485054
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4195910, 2.4272141
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6451039, 2.6393895
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4227605, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8627090, 1.8476133
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4369631, 1.4432813

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0347975, upper bound: 1.0409553
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0343680, upper bound: 1.0413927
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9469347, 2.9214945
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2047472, 2.2002344
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9576843, 1.9489853
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4199033, 2.4241116
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6461787, 2.6287632
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4244633, 2.4258060
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8445840, 1.8494463
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4365940, 1.4433156

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0349784, upper bound: 1.0406272
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0343764, upper bound: 1.0412293
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9430418, 2.9471371
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2040501, 2.2102363
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9626331, 1.9481368
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4222422, 2.4245629
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6421618, 2.6423316
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4275026, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8652759, 1.8450468
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4381661, 1.4420780

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370225, upper bound: 1.0387294
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0365930, upper bound: 1.0391668
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9454870, 2.9229417
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2050943, 2.1998875
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9580534, 1.9486170
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4225550, 2.4214602
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6432366, 2.6317050
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4210639
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8471503, 1.8468795
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4377973, 1.4421123

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0371951, upper bound: 1.0385582
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0367656, upper bound: 1.0389955
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9229422, 2.9672394
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1998873, 2.2143996
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9528844, 1.9580531
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4214602, 2.4253449
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6317048, 2.6527839
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4210639, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8631725, 1.8471508
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4424467, 1.4377972

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0389931, upper bound: 1.0367680
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0385557, upper bound: 1.0371975
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9214945, 2.9686866
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2002344, 2.2140527
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9532526, 1.9576845
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4241118, 2.4226935
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6287632, 2.6557260
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4258060, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8657393, 1.8445840
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4436498, 1.4365941

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0412269, upper bound: 1.0343788
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0406248, upper bound: 1.0349809
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9054465, 2.9111557
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1955490, 2.1953988
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8867819, 1.9205561
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3789673, 2.3419557
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6527872, 2.6512022
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8504646, 1.8509989
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4455068, 1.4401722

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0391724, upper bound: 1.0364311
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0413983, upper bound: 1.0342062
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8920541, 2.9245481
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1926279, 2.1983192
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9105289, 1.8968093
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3441367, 2.3767862
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1252232, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6388783, 2.6651111
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8496292, 1.8518348
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4456217, 1.4400573

Time for backsubstitution: 13.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0385724, upper bound: 1.0370332
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0407974, upper bound: 1.0348083
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9128680, 2.9160359
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1958976, 2.2028258
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8887987, 1.9308269
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3824539, 2.3461874
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6579099, 2.6528547
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8590713, 1.8513501
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4446568, 1.4416876

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0381264, upper bound: 1.0371953
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0403612, upper bound: 1.0349604
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8994756, 2.9294286
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1929774, 2.2057462
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9125462, 1.9070799
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3476233, 2.3810182
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6440010, 2.6667638
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8582358, 1.8521860
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4447718, 1.4415724

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395827, upper bound: 1.0377915
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0397552, upper bound: 1.0376202
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9345737, 2.9287856
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2067399, 2.2052975
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9541373, 1.9495749
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4104652, 2.4035587
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3271301
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6315742, 2.6213515
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.3924947, 2.4112110
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8330951, 1.8482649
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4420476, 1.4448166

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376977, upper bound: 1.0382334
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376919, upper bound: 1.0360066
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9315181, 2.9318423
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2086587, 2.2033787
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9544935, 1.9492190
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.4195533, 2.3944702
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3274283, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6237054, 2.6292205
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4082208, 2.3954849
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8411512, 1.8402088
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4466600, 1.4402039

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0377076, upper bound: 1.0359919
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0399326, upper bound: 1.0359977
time: 3.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0380552, upper bound: 1.0393282
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0374530, upper bound: 1.0399302
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0376177, upper bound: 1.0397577
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0370164, upper bound: 1.0403596
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0360000, upper bound: 1.0395947
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0382357, upper bound: 1.0373599
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0354000, upper bound: 1.0401967
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0376348, upper bound: 1.0379620
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0347975, upper bound: 1.0409553
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0343680, upper bound: 1.0413927
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0349784, upper bound: 1.0406272
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0343764, upper bound: 1.0412293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0370225, upper bound: 1.0387294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0365930, upper bound: 1.0391668
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0371951, upper bound: 1.0385582
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0367656, upper bound: 1.0389955
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0389931, upper bound: 1.0367680
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0385557, upper bound: 1.0371975
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0412269, upper bound: 1.0343788
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0406248, upper bound: 1.0349809
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0391724, upper bound: 1.0364311
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0413983, upper bound: 1.0342062
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0385724, upper bound: 1.0370332
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0407974, upper bound: 1.0348083
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0381264, upper bound: 1.0371953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0403612, upper bound: 1.0349604
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0395827, upper bound: 1.0377915
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0397552, upper bound: 1.0376202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0376977, upper bound: 1.0382334
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0376919, upper bound: 1.0360066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0377076, upper bound: 1.0359919
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.18
Output dim: 9, lower bound: -1.0399326, upper bound: 1.0359977

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9108367, 2.9373693
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1936507, 2.2133312
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8919451, 1.9314692
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3685317, 2.3625879
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6563172, 2.6629243
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8801882, 1.8446953
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4429450, 1.4436998

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0358183, upper bound: 1.0393271
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0380540, upper bound: 1.0370922
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.8974433, 2.9507623
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1907301, 2.2162516
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9156921, 1.9077225
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3337007, 2.3974185
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1240296, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6424084, 2.6768353
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8793523, 1.8455310
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4430599, 1.4435849

Time for backsubstitution: 13.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0352170, upper bound: 1.0399291
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0374518, upper bound: 1.0376943
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9139819, 2.9212255
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1958103, 2.2022810
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8957427, 1.9120660
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3709159, 2.3504138
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6581526, 2.6535523
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8684785, 1.8469863
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4419072, 1.4439030

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0353903, upper bound: 1.0397562
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0376162, upper bound: 1.0375311
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9005885, 2.9346185
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.1928902, 2.2052014
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9194896, 1.8883190
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3360848, 2.3852444
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6442437, 2.6674635
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8676426, 1.8478222
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4420223, 1.4437881

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0347898, upper bound: 1.0403581
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0370148, upper bound: 1.0381332
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.9125433, 2.8932366
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.2006402, 2.1997955
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8916967, 1.9174008
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.3469725, 2.3345282
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1211572
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.6460624, 2.6237276
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4187913, 2.4220040
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.8369563, 1.8411865
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.4403753, 1.4471840

Time for backsubstitution: 13.97 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=1.4499115943908691
rel_dist={9: [-1.041417153698955, 1.0414178245879198]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9248141, upper bound: 0.9264244
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264236, upper bound: 0.9248147
time: 3.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 9, lower bound: -0.9248141, upper bound: 0.9264244
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 9, lower bound: -0.9264236, upper bound: 0.9248147

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7067561, 2.7056308
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0468574, 2.0471272
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8024778, 1.8027647
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2750611, 2.2771237
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2173920, 2.2156534
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0419779, 2.0393536
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3829651, 2.3806767
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2543335, 2.2580216
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6875594, 1.6895556
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3506021, 1.3515379

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246506, upper bound: 0.9264219
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9248102, upper bound: 0.9262636
time: 3.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7056308, 2.7067561
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0471272, 2.0468574
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8027649, 1.8024781
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2771239, 2.2750614
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2156534, 2.2173920
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0393534, 2.0419781
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3806772, 2.3829651
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2580214, 2.2543333
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6895554, 1.6875594
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3515379, 1.3506021

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264211, upper bound: 0.9243563
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259540, upper bound: 0.9248119
time: 3.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.00
Output dim: 9, lower bound: -0.9246506, upper bound: 0.9264219
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.00
Output dim: 9, lower bound: -0.9248102, upper bound: 0.9262636
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.00
Output dim: 9, lower bound: -0.9264211, upper bound: 0.9243563
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.00
Output dim: 9, lower bound: -0.9259540, upper bound: 0.9248119

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7043109, 2.7220042
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0458131, 2.0541320
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8056903, 1.8022847
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2747493, 2.2792249
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2206268, 2.2151709
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0416803, 2.0413387
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3818898, 2.3878665
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2526302, 2.2694693
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6998243, 1.6877232
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3508549, 1.3515038

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246422, upper bound: 0.9259579
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241933, upper bound: 0.9264113
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7067561, 2.7031856
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0468574, 2.0460827
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8019981, 1.8027647
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2750611, 2.2768116
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2169094, 2.2156534
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0419779, 2.0390561
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3829651, 2.3796015
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2543335, 2.2563188
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6857266, 1.6895556
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3505681, 1.3515379

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9248018, upper bound: 0.9258000
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243529, upper bound: 0.9262553
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6802115, 2.6709204
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0415792, 2.0390379
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7419260, 1.7601094
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2195024, 2.1903498
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2250786, 2.2342560
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0155492, 2.0083818
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3925917, 2.3840616
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2794518, 2.2691965
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6879501, 1.6853034
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3517435, 1.3508972

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9243508
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264161, upper bound: 0.9218798
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6697946, 2.6813369
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0393081, 2.0413096
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7603958, 1.7416396
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1924114, 2.2174404
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2325172, 2.2268174
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0057573, 2.0181737
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3817737, 2.3948801
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2728848, 2.2757630
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6872997, 1.6859534
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3518331, 1.3508078

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243425, upper bound: 0.9248130
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259530, upper bound: 0.9248036
time: 3.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9246422, upper bound: 0.9259579
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9241933, upper bound: 0.9264113
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9248018, upper bound: 0.9258000
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9243529, upper bound: 0.9262553
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9239447, upper bound: 0.9243508
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9264161, upper bound: 0.9218798
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9243425, upper bound: 0.9248130
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -0.9259530, upper bound: 0.9248036

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7011676, 2.7314193
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0436540, 2.0605674
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8018932, 1.8135781
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2723646, 2.2863097
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2201500, 2.2165761
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0389957, 2.0493791
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3800550, 2.3933208
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2545009, 2.2688472
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7066407, 1.6854327
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3514588, 1.3513005

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246340, upper bound: 0.9259591
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246412, upper bound: 0.9243421
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7043109, 2.7188609
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0458131, 2.0519729
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8056903, 1.7984872
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2747493, 2.2768407
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2206268, 2.2146940
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0416803, 2.0386541
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3818898, 2.3860316
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2520084, 2.2694693
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6975341, 1.6877232
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3506515, 1.3515038

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217246, upper bound: 0.9264066
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241883, upper bound: 0.9239371
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7036133, 2.7126007
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0446987, 2.0525184
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7982006, 1.8140583
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2726774, 2.2838964
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2164326, 2.2170587
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0392928, 2.0470963
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3811293, 2.3850558
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2562037, 2.2556968
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6925426, 1.6872654
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3511717, 1.3513348

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9247992, upper bound: 0.9250055
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9240058, upper bound: 0.9257972
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7067561, 2.7000420
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0468574, 2.0439239
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8019981, 1.7989671
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2750611, 2.2744274
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2169094, 2.2151766
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0419779, 2.0363715
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3829651, 2.3777666
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2537117, 2.2563188
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6834359, 1.6895556
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3503647, 1.3515379

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243502, upper bound: 0.9257839
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238899, upper bound: 0.9262510
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6712008, 2.6533692
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0390921, 2.0355339
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7516694, 1.7665758
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1993856, 2.1762528
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2249064, 2.2343187
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0206838, 2.0121253
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3912373, 2.3776982
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2755671, 2.2637315
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6838460, 1.6795042
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3437994, 1.3458813

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237896, upper bound: 0.9243465
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239404, upper bound: 0.9241885
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6626601, 2.6619096
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0380754, 2.0365505
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7483926, 1.7698526
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2054052, 2.1702328
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2251410, 2.2340841
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0192924, 2.0135167
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3862281, 2.3827074
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2739868, 2.2653120
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6821504, 1.6811998
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3467276, 1.3429528

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264135, upper bound: 0.9210810
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9256226, upper bound: 0.9218790
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6691809, 2.6783454
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0430937, 2.0465879
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7610986, 1.7426188
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1681571, 2.2002537
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2173138, 2.2053547
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0178018, 2.0222981
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3668594, 2.3738456
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2309465, 2.2460563
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6820512, 1.6869709
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3553882, 1.3579504

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243341, upper bound: 0.9243537
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238810, upper bound: 0.9248026
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6668034, 2.6807456
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0445862, 2.0450952
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7613752, 1.7423420
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1752305, 2.1931851
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2110548, 2.2116132
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0098815, 2.0302179
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3607392, 2.3799853
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2431793, 2.2338247
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6883168, 1.6807048
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3589780, 1.3543630

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259504, upper bound: 0.9240083
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251566, upper bound: 0.9248032
time: 3.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9246340, upper bound: 0.9259591
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9246412, upper bound: 0.9243421
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9217246, upper bound: 0.9264066
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9241883, upper bound: 0.9239371
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9247992, upper bound: 0.9250055
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9240058, upper bound: 0.9257972
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9243502, upper bound: 0.9257839
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9238899, upper bound: 0.9262510
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9237896, upper bound: 0.9243465
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9239404, upper bound: 0.9241885
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9264135, upper bound: 0.9210810
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9256226, upper bound: 0.9218790
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9243341, upper bound: 0.9243537
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9238810, upper bound: 0.9248026
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9259504, upper bound: 0.9240083
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.71
Output dim: 9, lower bound: -0.9251566, upper bound: 0.9248032

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.7005768, 2.7284284
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0474386, 2.0658448
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8025951, 1.8145580
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2481108, 2.2691293
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2049465, 2.1951141
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0510402, 2.0535033
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3651590, 2.3722854
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2125621, 2.2391410
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7013919, 1.6864495
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3550138, 1.3584456

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221653, upper bound: 0.9259541
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246290, upper bound: 0.9234826
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6981764, 2.7308059
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0489311, 2.0643523
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8028727, 1.8142810
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2551799, 2.2620554
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1986880, 2.2013731
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0431204, 2.0614233
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3590193, 2.3784058
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2247939, 2.2269087
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7076576, 1.6801839
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3586013, 1.3548558

Time for backsubstitution: 13.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221725, upper bound: 0.9243371
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246363, upper bound: 0.9218637
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6953001, 2.7013102
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0433254, 2.0484686
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8154337, 1.8049531
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2546325, 2.2627437
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2204537, 2.2147565
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0468163, 2.0423975
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3805346, 2.3796670
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2481232, 2.2640038
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6934304, 1.6819243
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3427076, 1.3464886

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217163, upper bound: 0.9264075
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217236, upper bound: 0.9247837
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6867595, 2.7098520
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0423088, 2.0494857
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8121564, 1.8082299
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2606521, 2.2567239
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2206883, 2.2145219
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0454249, 2.0437889
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3755250, 2.3846731
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2465429, 2.2655854
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6917348, 1.6836197
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3456364, 1.3435601

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241857, upper bound: 0.9231428
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233922, upper bound: 0.9239327
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6905661, 2.7077703
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0415969, 2.0513682
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7916288, 1.8116357
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2654376, 2.2812223
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2162151, 2.2164693
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0321398, 2.0444577
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3725462, 2.3818843
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2560554, 2.2552872
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6913319, 1.6839981
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3503923, 1.3492191

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223231, upper bound: 0.9250005
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9247942, upper bound: 0.9225367
time: 3.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6987848, 2.6995540
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0435481, 2.0494170
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7957778, 1.8074861
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2700038, 2.2766566
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2158422, 2.2168417
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0366554, 2.0399432
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3779578, 2.3764725
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2557940, 2.2555480
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6892757, 1.6860535
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3490555, 1.3505557

Time for backsubstitution: 13.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9240032, upper bound: 0.9253269
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9235440, upper bound: 0.9257946
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6813369, 2.6642067
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0413098, 2.0361047
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7411599, 1.7565987
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2174397, 2.1897154
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2263336, 2.2320409
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0181737, 2.0027747
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3948801, 2.3788633
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2751403, 2.2711828
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6818304, 1.6872997
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3505702, 1.3518330

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9218741, upper bound: 0.9257786
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243452, upper bound: 0.9233149
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6709204, 2.6746233
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0390382, 2.0383761
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7596292, 1.7381289
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1903496, 2.2168057
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2337723, 2.2246020
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0083818, 2.0125668
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3840621, 2.3896813
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2685738, 2.2777493
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6811805, 1.6879497
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3506596, 1.3517436

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238816, upper bound: 0.9262518
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238888, upper bound: 0.9246289
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6687555, 2.6697431
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0380473, 2.0425386
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7548823, 1.7660959
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1990728, 2.1783535
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2281418, 2.2338362
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0203867, 2.0141103
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3901625, 2.3848879
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2738643, 2.2751791
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6961105, 1.6776714
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3440521, 1.3458472

Time for backsubstitution: 14.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237811, upper bound: 0.9238855
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233278, upper bound: 0.9243385
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6712008, 2.6509244
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0390921, 2.0344894
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7511897, 1.7665758
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1993856, 2.1759405
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2244244, 2.2343187
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0206838, 2.0118277
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3912373, 2.3766229
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2755671, 2.2620285
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6820133, 1.6795042
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3437650, 1.3458813

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9239319, upper bound: 0.9237271
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234786, upper bound: 0.9241793
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6496158, 2.6570840
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0349741, 2.0354009
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7418208, 1.7674298
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1981659, 2.1675591
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2249241, 2.2334948
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0121403, 2.0108802
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3776450, 2.3795362
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2738380, 2.2649024
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6809394, 1.6779332
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3459487, 1.3408369

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9262511, upper bound: 0.9210768
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264091, upper bound: 0.9209262
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6578345, 2.6488652
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0369253, 2.0334496
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7459698, 1.7632806
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2027321, 2.1629932
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2245522, 2.2338672
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0166559, 2.0063646
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3830566, 2.3741241
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2735767, 2.2651634
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6788843, 1.6799889
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3446116, 1.3421739

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9254607, upper bound: 0.9218727
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9256182, upper bound: 0.9217216
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6660376, 2.6877611
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0409341, 2.0530231
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7573016, 1.7539124
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1657729, 2.2073393
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2168374, 2.2067604
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0151162, 2.0303376
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3650231, 2.3792987
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2328167, 2.2454336
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6888666, 1.6846797
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3559921, 1.3577473

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9218574, upper bound: 0.9243507
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9243291, upper bound: 0.9218796
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6691809, 2.6752024
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0430937, 2.0444286
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7610986, 1.7388215
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1681571, 2.1978700
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2173138, 2.2048783
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0178018, 2.0196126
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3668594, 2.3720095
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2303243, 2.2460563
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6797600, 1.6869709
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3551848, 1.3579504

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9237188, upper bound: 0.9247986
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9238766, upper bound: 0.9246389
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6537585, 2.6759200
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0414853, 2.0439456
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7548029, 1.7399187
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1679912, 2.1905119
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2108374, 2.2110233
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0027299, 2.0275817
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3521562, 2.3768141
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2430310, 2.2334156
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6871061, 1.6774385
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3581991, 1.3522472

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5790

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9234737, upper bound: 0.9240033
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9259454, upper bound: 0.9215322
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6619778, 2.6677012
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0434365, 2.0419946
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7589519, 1.7357695
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1725574, 2.1859460
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2104654, 2.2113957
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0072455, 2.0230658
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3575678, 2.3714023
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2427702, 2.2336767
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6850505, 1.6794939
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3568625, 1.3535841

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9251479, upper bound: 0.9243440
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246937, upper bound: 0.9247948
time: 3.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9221653, upper bound: 0.9259541
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9246290, upper bound: 0.9234826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9221725, upper bound: 0.9243371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9246363, upper bound: 0.9218637
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9217163, upper bound: 0.9264075
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9217236, upper bound: 0.9247837
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9241857, upper bound: 0.9231428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9233922, upper bound: 0.9239327
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9223231, upper bound: 0.9250005
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9247942, upper bound: 0.9225367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9240032, upper bound: 0.9253269
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9235440, upper bound: 0.9257946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9218741, upper bound: 0.9257786
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9243452, upper bound: 0.9233149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9238816, upper bound: 0.9262518
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9238888, upper bound: 0.9246289
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9237811, upper bound: 0.9238855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9233278, upper bound: 0.9243385
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9239319, upper bound: 0.9237271
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9234786, upper bound: 0.9241793
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9262511, upper bound: 0.9210768
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9264091, upper bound: 0.9209262
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9254607, upper bound: 0.9218727
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9256182, upper bound: 0.9217216
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9218574, upper bound: 0.9243507
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9243291, upper bound: 0.9218796
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9237188, upper bound: 0.9247986
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9238766, upper bound: 0.9246389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9234737, upper bound: 0.9240033
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9259454, upper bound: 0.9215322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9251479, upper bound: 0.9243440
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.65
Output dim: 9, lower bound: -0.9246937, upper bound: 0.9247948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6915660, 2.7108777
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0449510, 2.0623405
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8123386, 1.8210239
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2279935, 2.2550316
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2047744, 2.1951766
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0561743, 2.0572462
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3638048, 2.3659220
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2086773, 2.2336755
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6972876, 1.6806505
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3470700, 1.3534300

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221627, upper bound: 0.9251592
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213693, upper bound: 0.9259515
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6830258, 2.7194195
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0439343, 2.0633576
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8090618, 1.8243008
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2340136, 2.2490118
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.2050090, 2.1949420
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0547829, 2.0586376
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3587956, 2.3709278
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2070961, 2.2352571
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6955929, 1.6823461
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3499982, 1.3505018

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246264, upper bound: 0.9230100
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9241712, upper bound: 0.9234800
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6891661, 2.7132552
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0464435, 2.0608480
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8126156, 1.8207474
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2350626, 2.2479579
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1985154, 2.2014356
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0482550, 2.0651662
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3576651, 2.3720422
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2209082, 2.2214429
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7035542, 1.6743848
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3506572, 1.3498402

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221699, upper bound: 0.9238741
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9217146, upper bound: 0.9243324
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.6806259, 2.7217970
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.0454268, 2.0618651
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.8093383, 1.8240242
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.2410822, 2.2419379
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1987500, 2.2012010
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.0468626, 2.0665576
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.3526559, 2.3770483
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.2193279, 2.2230246
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.7018585, 1.6760802
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3535860, 1.3469117

Time for backsubstitution: 14.16 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=1.353825569152832
rel_dist={9: [-0.9264248152800034, 0.9264272454017544]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5790
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5790

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658477, upper bound: 0.8660059
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660033, upper bound: 0.8658487
time: 3.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.02
Output dim: 9, lower bound: -0.8658477, upper bound: 0.8660059
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.02
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

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658450, upper bound: 0.8655691
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654145, upper bound: 0.8660014
time: 6.74 seconds

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

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8638811, upper bound: 0.8658443
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659990, upper bound: 0.8637264
time: 3.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 9, lower bound: -0.8658450, upper bound: 0.8655691
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 9, lower bound: -0.8654145, upper bound: 0.8660014
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 9, lower bound: -0.8638811, upper bound: 0.8658443
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 9, lower bound: -0.8659990, upper bound: 0.8637264

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5557752, 2.5629773
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9591169, 1.9640694
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6706851, 1.6833515
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1371698, 2.1160178
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1698570, 2.1730471
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9658952, 1.9594586
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2658591, 2.2636707
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1987648, 2.2044077
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6200793, 1.6074381
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3061993, 1.3060300

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658427, upper bound: 0.8648848
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8651561, upper bound: 0.8655667
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5468469, 2.5719056
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9571695, 1.9660163
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6865165, 1.6675203
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1139498, 2.1392384
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1762342, 2.1666710
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9575019, 1.9678516
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2565866, 2.2729449
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1931362, 2.2100363
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6195223, 1.6079953
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3062761, 1.3059535

Time for backsubstitution: 13.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654072, upper bound: 0.8655087
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8649258, upper bound: 0.8659940
time: 7.31 seconds

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

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8620482, upper bound: 0.8658451
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8638802, upper bound: 0.8640115
time: 3.89 seconds

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
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8641661, upper bound: 0.8637272
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659980, upper bound: 0.8618935
time: 3.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8658427, upper bound: 0.8648848
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8651561, upper bound: 0.8655667
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8654072, upper bound: 0.8655087
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8649258, upper bound: 0.8659940
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8620482, upper bound: 0.8658451
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8638802, upper bound: 0.8640115
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8641661, upper bound: 0.8637272
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 9, lower bound: -0.8659980, upper bound: 0.8618935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5427318, 2.5569782
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9560156, 1.9626405
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6641128, 1.6803353
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1299305, 2.1126919
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1695862, 2.1724572
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9587431, 1.9561777
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2572756, 2.2597256
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1985784, 2.2039981
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6185741, 1.6041720
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3052292, 1.3039143

Time for backsubstitution: 14.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8637206, upper bound: 0.8648805
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658385, upper bound: 0.8627626
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5497766, 2.5499339
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9576883, 1.9609680
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6676691, 1.6767788
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1338444, 2.1087782
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1692677, 2.1727765
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9626141, 1.9523070
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2619143, 2.2550869
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1983547, 2.2042217
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6168127, 1.6059337
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3040833, 1.3050599

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8630339, upper bound: 0.8655646
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8651519, upper bound: 0.8634467
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5437045, 2.5795276
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9550104, 1.9712236
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6827190, 1.6766582
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1115646, 2.1449704
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1757579, 2.1678081
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9548168, 1.9743595
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2547507, 2.2773571
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1946497, 2.2094135
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6250362, 1.6057045
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3067651, 1.3057506

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8636111, upper bound: 0.8655095
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654058, upper bound: 0.8636877
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5468469, 2.5687633
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9571695, 1.9638569
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6865165, 1.6637230
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1139498, 2.1368539
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1762342, 2.1661949
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9575019, 1.9651668
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2565866, 2.2711091
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1925130, 2.2100363
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6172314, 1.6079953
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3060732, 1.3059535

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8628036, upper bound: 0.8659895
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8649215, upper bound: 0.8638736
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5699959, 2.5581923
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9671841, 1.9665484
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7383358, 1.7362447
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1546030, 2.1655095
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1409760, 2.1362956
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9935451, 1.9852664
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2387052, 2.2280912
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1339231, 2.1413503
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5896595, 1.5954096
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2964380, 1.3020580

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8620459, upper bound: 0.8651562
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8613585, upper bound: 0.8658429
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5606380, 2.5675507
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9675922, 1.9661403
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7357647, 1.7388160
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1658220, 2.1542904
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1358128, 2.1414592
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9855638, 1.9932475
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2291656, 2.2376311
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1430521, 2.1322207
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5935771, 1.5914922
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3020232, 1.2964728

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659954, upper bound: 0.8614917
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655634, upper bound: 0.8618908
time: 4.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8637206, upper bound: 0.8648805
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8658385, upper bound: 0.8627626
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8630339, upper bound: 0.8655646
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8651519, upper bound: 0.8634467
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8636111, upper bound: 0.8655095
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8654058, upper bound: 0.8636877
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8628036, upper bound: 0.8659895
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8649215, upper bound: 0.8638736
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8620459, upper bound: 0.8651562
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8613585, upper bound: 0.8658429
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8659954, upper bound: 0.8614917
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 9, lower bound: -0.8655634, upper bound: 0.8618908

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5251799, 2.5467477
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9525118, 1.9600079
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6705792, 1.6896105
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1149740, 2.0925760
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1696157, 2.1722858
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9624858, 1.9611127
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2509122, 2.2576535
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1931124, 2.1998882
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6127756, 1.5998261
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2997956, 1.2959704

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8640173, upper bound: 0.8627613
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658371, upper bound: 0.8609654
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5395446, 2.5323815
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9550552, 1.9574640
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6769445, 1.6832454
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1137280, 2.0938220
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1690960, 2.1728063
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9675498, 1.9560497
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2598443, 2.2487237
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1942434, 2.1987562
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6124666, 1.6001346
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2961397, 1.2996264

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8612125, upper bound: 0.8655614
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8630326, upper bound: 0.8637700
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5322242, 2.5397031
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9541836, 1.9583356
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6741354, 1.6860540
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1188879, 2.0886621
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1692967, 2.1726050
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9663568, 1.9572423
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2555509, 2.2530146
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1928892, 2.2001119
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6110141, 1.6015880
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2986500, 1.2971163

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8633181, upper bound: 0.8634458
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8651509, upper bound: 0.8616450
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5407190, 2.5755777
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9554954, 1.9719396
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6832361, 1.6774209
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1044893, 2.1396623
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1712971, 2.1618578
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9480805, 1.9653730
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2487700, 2.2694149
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1820221, 2.1999474
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6181984, 1.6005776
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3035417, 1.3033293

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8636023, upper bound: 0.8655062
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8636101, upper bound: 0.8636702
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5397539, 2.5765424
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9557261, 1.9717083
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6834817, 1.6771753
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1062565, 2.1378946
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1698074, 2.1633480
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9458308, 1.9676228
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2468088, 2.2713764
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1851835, 2.1967859
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6199098, 1.5988665
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3043437, 1.3025273

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4628
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4628

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8632836, upper bound: 0.8636852
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654016, upper bound: 0.8615672
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5366158, 2.5512118
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9545364, 1.9603524
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6957915, 1.6701896
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0938330, 2.1218979
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1760612, 2.1662233
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9624376, 1.9689097
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2545166, 2.2647455
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1884022, 2.2045698
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6128850, 1.6021960
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2981291, 1.3005199

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8628014, upper bound: 0.8653025
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8621172, upper bound: 0.8659877
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5569515, 2.5521936
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9640841, 1.9651198
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7317641, 1.7332287
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1473632, 2.1621835
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1407056, 2.1357062
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9863925, 1.9819846
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2301226, 2.2241473
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1337376, 2.1409411
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5881548, 1.5921435
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2954681, 1.2999415

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8620434, upper bound: 0.8647256
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8616424, upper bound: 0.8651536
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5639963, 2.5451488
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9657564, 1.9634476
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.7353203, 1.7296724
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1512766, 2.1582699
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1403861, 2.1360254
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9902625, 1.9781139
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2347612, 2.2195084
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1335139, 2.1411648
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5863934, 1.5939052
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2943223, 1.3010874

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6166
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8613560, upper bound: 0.8654077
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609557, upper bound: 0.8658402
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8641697, upper bound: 0.8614864
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8659940, upper bound: 0.8614901
time: 3.95 seconds

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

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8655611, upper bound: 0.8612011
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648771, upper bound: 0.8618902
time: 4.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8640173, upper bound: 0.8627613
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8658371, upper bound: 0.8609654
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8612125, upper bound: 0.8655614
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8630326, upper bound: 0.8637700
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8633181, upper bound: 0.8634458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8651509, upper bound: 0.8616450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8636023, upper bound: 0.8655062
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8636101, upper bound: 0.8636702
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8632836, upper bound: 0.8636852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8654016, upper bound: 0.8615672
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8628014, upper bound: 0.8653025
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8621172, upper bound: 0.8659877
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8620434, upper bound: 0.8647256
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8616424, upper bound: 0.8651536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8613560, upper bound: 0.8654077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8609557, upper bound: 0.8658402
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8641697, upper bound: 0.8614864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8659940, upper bound: 0.8614901
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8655611, upper bound: 0.8612011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 9, lower bound: -0.8648771, upper bound: 0.8618902

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5212302, 2.5437632
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9532266, 1.9604919
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6713419, 1.6901278
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1096649, 2.0854993
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1636667, 2.1678262
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9534998, 1.9543765
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2429695, 2.2516720
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1836462, 2.1872606
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6076484, 1.5929875
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2973742, 1.2927471

Time for backsubstitution: 14.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8640002, upper bound: 0.8609643
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8658362, upper bound: 0.8609567
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5365596, 2.5284321
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9555392, 1.9581790
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6774616, 1.6840084
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1066513, 2.0885131
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1646366, 2.1668561
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9608135, 1.9470637
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2538633, 2.2407811
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1816158, 2.1892900
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6056280, 1.5950072
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2929163, 1.2972051

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8611985, upper bound: 0.8655602
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8612063, upper bound: 0.8637575
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5252852, 2.5348024
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9592481, 1.9621203
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6750743, 1.6867561
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.1006913, 2.0644064
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1478348, 2.1565077
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9567819, 1.9544561
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2345166, 2.2372262
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1614327, 2.1581709
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6006963, 1.5858994
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3003585, 1.2957503

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8633242, upper bound: 0.8616398
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8651496, upper bound: 0.8616433
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5397849, 2.5725861
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9592805, 1.9770038
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6839385, 1.6783605
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0802350, 2.1214707
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1552005, 2.1403956
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9589934, 1.9694974
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2329984, 2.2483807
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1400838, 2.1684942
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6129496, 1.6006992
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3070972, 1.3099618

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4670
type: RSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8636001, upper bound: 0.8648175
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8629132, upper bound: 0.8655036
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5222030, 2.5663128
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9522219, 1.9690752
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6899476, 1.6864498
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0913000, 2.1177778
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1698380, 2.1631773
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9495745, 1.9725592
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2404447, 2.2693031
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1797180, 2.1926763
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6141105, 1.5945201
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2989101, 1.2945832

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 4670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8635945, upper bound: 0.8615612
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8654007, upper bound: 0.8615534
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5235715, 2.5452099
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9514360, 1.9589233
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6892192, 1.6671731
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0865936, 2.1185720
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1757917, 2.1656349
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9552855, 1.9656274
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2459331, 2.2608008
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1882172, 2.2041609
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6113799, 1.5989301
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2971592, 1.2984040

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609945, upper bound: 0.8653002
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8628005, upper bound: 0.8634690
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5306158, 2.5381651
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9531078, 1.9572511
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6927760, 1.6636169
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0905070, 2.1146581
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1754727, 2.1659541
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9591565, 1.9617569
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2505717, 2.2561622
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1879935, 2.2043846
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.6096184, 1.6006918
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2960134, 1.2995498

Time for backsubstitution: 14.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5799
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5799

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8603069, upper bound: 0.8659868
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8621163, upper bound: 0.8641567
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5211153, 2.5252857
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9562650, 1.9592481
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6867559, 1.6723895
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0626497, 2.1006911
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1565075, 2.1451321
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9527965, 1.9567819
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2312198, 2.2345169
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1486011, 2.1614320
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5858991, 1.5904448
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2957504, 1.3001474

Time for backsubstitution: 13.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8616407, upper bound: 0.8651523
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8616371, upper bound: 0.8633268
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5370879, 2.5093126
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9598842, 1.9556286
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6744812, 1.6846642
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0897846, 2.0735569
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1498127, 2.1518273
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9650602, 1.9445183
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2451310, 2.2206051
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1540055, 2.1560273
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5846946, 1.5916493
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2945278, 1.3013698

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8613544, upper bound: 0.8654062
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8613508, upper bound: 0.8636135
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5281596, 2.5182409
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9579377, 1.9575758
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6903121, 1.6688333
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0665636, 2.0967774
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1561880, 2.1454513
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9566674, 1.9529114
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2358584, 2.2298782
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1483769, 2.1616557
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5841377, 1.5922065
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.2946045, 1.3012933

Time for backsubstitution: 13.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8609483, upper bound: 0.8653479
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8604628, upper bound: 0.8658313
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.6260853, -9.4460907, -12.6260853, -9.4460907, -2.5337291, 2.5326984
1: -11.7355261, -9.1776123, -11.7355261, -9.1776123, -1.9624367, 1.9588065
2: -8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.6756887, 1.6943257
3: -7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.0990252, 2.0625014
4: -3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.1392884, 2.1528010
5: -5.9543295, -3.8286073, -5.9543295, -3.8286073, -1.9650731, 1.9666142
6: -16.9029446, -13.7977066, -16.9029446, -13.7977066, -2.2315938, 2.2327640
7: -4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.1540818, 2.1344595
8: -5.2317653, -2.9253664, -5.2317653, -2.9253664, -1.5971899, 1.5928376
9: 4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.3047311, 1.2984526

Time for backsubstitution: 13.97 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2423.78 seconds
