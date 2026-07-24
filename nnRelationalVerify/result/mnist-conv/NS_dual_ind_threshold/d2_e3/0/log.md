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
execution time: IAR + RelationalAnalysis = 20.95 + 34.57 = 55.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8656694, upper bound: 0.8656683

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656619, upper bound: 0.8630922
time: 5.74 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656621, upper bound: 0.8656605
time: 4.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.42
Output dim: 8, lower bound: -0.8656619, upper bound: 0.8630922
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.42
Output dim: 8, lower bound: -0.8656621, upper bound: 0.8656605

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.0024648, -4.3908353, -7.0047169, -4.3822222, -1.8244171, 1.8461704
1: -9.1236200, -6.9862585, -9.1257458, -6.9854612, -1.7083998, 1.7096744
2: -7.6828299, -5.9806924, -7.6841359, -5.9738283, -1.4483485, 1.4605088
3: -5.6804662, -3.6361558, -5.6812067, -3.6321764, -1.9539042, 1.9594889
4: -9.2557974, -7.2503557, -9.2624378, -7.2493510, -1.6724272, 1.6737976
5: 1.3473225, 2.7011001, 1.3430867, 2.7020378, -1.1858101, 1.1701021
6: -1.6131902, 0.3668239, -1.6151254, 0.3804488, -1.3818045, 1.3969998
7: -10.3694105, -8.7651758, -10.3734074, -8.7645588, -1.3322654, 1.3195477
8: 5.5951490, 7.2538667, 5.5920753, 7.2585230, -1.6633739, 1.6617913
9: -5.3400903, -3.8939803, -5.3470459, -3.8933206, -1.2847166, 1.2725477

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630933, upper bound: 0.8630920
time: 6.01 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630924, upper bound: 0.8630923
time: 5.52 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.0502157, -4.3795662, -7.0050087, -4.3816023, -1.8958340, 1.8573942
1: -9.1307707, -6.9772511, -9.1259117, -6.9853792, -1.7164097, 1.7181554
2: -7.7209396, -5.9688120, -7.6842713, -5.9733448, -1.4851017, 1.4727783
3: -5.6883774, -3.6148036, -5.6812730, -3.6317525, -1.9628735, 1.9718142
4: -9.2687082, -7.2111197, -9.2628956, -7.2492146, -1.6857786, 1.7013283
5: 1.3402802, 2.7186675, 1.3427935, 2.7021685, -1.1931276, 1.2130551
6: -1.6832242, 0.3850594, -1.6153748, 0.3813987, -1.4433455, 1.4153171
7: -10.3848066, -8.7405834, -10.3737068, -8.7644348, -1.3470807, 1.3577509
8: 5.5665436, 7.2632270, 5.5918055, 7.2588544, -1.6923108, 1.6714215
9: -5.3557873, -3.8577611, -5.3475370, -3.8932202, -1.3005505, 1.3065283

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8643642
time: 4.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8656593
time: 4.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.17 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.17
Output dim: 8, lower bound: -0.8630933, upper bound: 0.8630920
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 30.17
Output dim: 8, lower bound: -0.8630924, upper bound: 0.8630923
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8643642
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8656593

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.0483522, -4.3902607, -6.9923406, -4.4045591, -1.8714275, 1.8346252
1: -9.1268787, -6.9777589, -9.1165123, -6.9893880, -1.7082806, 1.7085967
2: -7.7184315, -5.9692249, -7.6779871, -5.9761310, -1.4783149, 1.4660525
3: -5.6810465, -3.6160231, -5.6649914, -3.6399539, -1.9476280, 1.9546118
4: -9.2677336, -7.2119231, -9.2607632, -7.2512670, -1.6812592, 1.6967554
5: 1.3417952, 2.7147775, 1.3492861, 2.6939001, -1.1838756, 1.2017531
6: -1.6823182, 0.3828030, -1.6115861, 0.3762805, -1.4372663, 1.4087896
7: -10.3841524, -8.7436180, -10.3699293, -8.7711201, -1.3397942, 1.3494713
8: 5.5702791, 7.2623529, 5.6002011, 7.2536721, -1.6833930, 1.6621518
9: -5.3538647, -3.8580251, -5.3432584, -3.8945098, -1.2963629, 1.3022087

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8643627
time: 4.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8643628
time: 4.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.0502138, -4.3795710, -7.0050097, -4.3816061, -1.8818607, 1.8573914
1: -9.1307707, -6.9772501, -9.1259108, -6.9853802, -1.7164087, 1.7177067
2: -7.7209392, -5.9688125, -7.6842685, -5.9733458, -1.4836621, 1.4727216
3: -5.6883745, -3.6148024, -5.6812687, -3.6317539, -1.9628716, 1.9713430
4: -9.2687092, -7.2111220, -9.2628937, -7.2492161, -1.6832809, 1.7041669
5: 1.3402809, 2.7186668, 1.3427944, 2.7021666, -1.1927371, 1.2099519
6: -1.6832235, 0.3850586, -1.6153738, 0.3813958, -1.4426599, 1.4161968
7: -10.3848076, -8.7405853, -10.3737049, -8.7644367, -1.3435845, 1.3557155
8: 5.5665445, 7.2632270, 5.5918074, 7.2588549, -1.6923103, 1.6714196
9: -5.3557868, -3.8577609, -5.3475351, -3.8932209, -1.2989039, 1.3084235

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8656594
time: 4.58 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8656594
time: 4.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.04 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.04
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8643627
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 31.04
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8643628
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.04
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8656594
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.04
Output dim: 8, lower bound: -0.8643632, upper bound: 0.8656594

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0374994, -4.4025249, -7.0050097, -4.3816061, -1.8669958, 1.8346024
1: -9.1213655, -6.9812460, -9.1259108, -6.9853802, -1.7074776, 1.7139454
2: -7.7146554, -5.9715834, -7.6842685, -5.9733458, -1.4768581, 1.4695454
3: -5.6720815, -3.6229658, -5.6812687, -3.6317539, -1.9466743, 1.9634299
4: -9.2664223, -7.2131619, -9.2628937, -7.2492161, -1.6800766, 1.6956453
5: 1.3468397, 2.7103770, 1.3427944, 2.7021666, -1.1876469, 1.2016497
6: -1.6794548, 0.3799219, -1.6153738, 0.3813958, -1.4361281, 1.4099493
7: -10.3810883, -8.7472782, -10.3737049, -8.7644367, -1.3433547, 1.3490901
8: 5.5749202, 7.2580876, 5.5918074, 7.2588549, -1.6839347, 1.6662803
9: -5.3514342, -3.8590455, -5.3475351, -3.8932209, -1.2947288, 1.3023629

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8617939, upper bound: 0.8656590
time: 4.73 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8656597
time: 5.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0502138, -4.3795743, -7.0050097, -4.3816061, -1.8874617, 1.8451338
1: -9.1307697, -6.9772487, -9.1259108, -6.9853802, -1.7159615, 1.7177067
2: -7.7209368, -5.9688120, -7.6842685, -5.9733458, -1.4840078, 1.4733300
3: -5.6883712, -3.6148036, -5.6812687, -3.6317539, -1.9624014, 1.9713440
4: -9.2687073, -7.2111220, -9.2628937, -7.2492161, -1.6905718, 1.7083044
5: 1.3402811, 2.7186656, 1.3427944, 2.7021666, -1.1927381, 1.2096953
6: -1.6832237, 0.3850563, -1.6153738, 0.3813958, -1.4425206, 1.4161959
7: -10.3848076, -8.7405872, -10.3737049, -8.7644367, -1.3435845, 1.3533878
8: 5.5665455, 7.2632251, 5.5918074, 7.2588549, -1.6923094, 1.6714177
9: -5.3557844, -3.8577619, -5.3475351, -3.8932209, -1.3038092, 1.3126044

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8643621
time: 8.28 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8643627
time: 9.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 39.89 seconds
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 39.89
Output dim: 8, lower bound: -0.8617939, upper bound: 0.8656590
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 39.89
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8656597
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 39.89
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8643621
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 39.89
Output dim: 8, lower bound: -0.8617929, upper bound: 0.8643627

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.0373926, -4.4025278, -7.0024652, -4.3908377, -1.8576355, 1.8039732
1: -9.1213608, -6.9812560, -9.1236172, -6.9862590, -1.7060127, 1.7111878
2: -7.7146034, -5.9715853, -7.6828289, -5.9806938, -1.4677539, 1.4501934
3: -5.6720743, -3.6230166, -5.6804600, -3.6361580, -1.9471502, 1.9618797
4: -9.2664175, -7.2132025, -9.2557964, -7.2503567, -1.6746082, 1.6867023
5: 1.3468409, 2.7103124, 1.3473229, 2.7010994, -1.1671495, 1.1925392
6: -1.6793177, 0.3799162, -1.6131909, 0.3668208, -1.4213390, 1.3808322
7: -10.3810759, -8.7473221, -10.3694115, -8.7651777, -1.3262873, 1.3420594
8: 5.5749536, 7.2580857, 5.5951519, 7.2538681, -1.6789145, 1.6629338
9: -5.3514304, -3.8591101, -5.3400879, -3.8939795, -1.2751827, 1.2933311

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8617916, upper bound: 0.8642063
time: 4.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8656550
time: 4.27 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.0376301, -4.4025211, -7.0503387, -4.3795691, -1.8688154, 1.8664188
1: -9.1213703, -6.9812365, -9.1307745, -6.9772372, -1.7135391, 1.7182574
2: -7.7147174, -5.9715815, -7.7209978, -5.9688101, -1.4815745, 1.4838643
3: -5.6720881, -3.6229057, -5.6883793, -3.6147442, -1.9623737, 1.9690928
4: -9.2664251, -7.2131138, -9.2687140, -7.2110744, -1.7016020, 1.7018666
5: 1.3468372, 2.7104511, 1.3402795, 2.7187390, -1.2047663, 1.2043791
6: -1.6796160, 0.3799267, -1.6833820, 0.3850634, -1.4398170, 1.4399981
7: -10.3811016, -8.7472286, -10.3848209, -8.7405357, -1.3600545, 1.3589835
8: 5.5748816, 7.2580934, 5.5665064, 7.2632279, -1.6883464, 1.6915870
9: -5.3514409, -3.8589721, -5.3557916, -3.8576891, -1.3084531, 1.3104289

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8642057
time: 4.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8656557
time: 4.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 32.10 seconds
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 32.10
Output dim: 8, lower bound: -0.8617916, upper bound: 0.8642063
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.10
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8656550
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 32.10
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8642057
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.10
Output dim: 8, lower bound: -0.8617906, upper bound: 0.8656557

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.0373912, -4.4025269, -7.0024652, -4.3908377, -1.8575039, 1.8036323
1: -9.1213589, -6.9812555, -9.1236172, -6.9862590, -1.7060127, 1.7115135
2: -7.7146020, -5.9715862, -7.6828289, -5.9806938, -1.4674997, 1.4501920
3: -5.6720743, -3.6230185, -5.6804600, -3.6361580, -1.9471493, 1.9579048
4: -9.2664137, -7.2132049, -9.2557964, -7.2503567, -1.6654024, 1.6853848
5: 1.3468416, 2.7103109, 1.3473229, 2.7010994, -1.1671495, 1.1915169
6: -1.6793177, 0.3799171, -1.6131909, 0.3668208, -1.4208918, 1.3781104
7: -10.3810759, -8.7473221, -10.3694115, -8.7651777, -1.3208642, 1.3411975
8: 5.5749550, 7.2580814, 5.5951519, 7.2538681, -1.6789131, 1.6629295
9: -5.3514280, -3.8591099, -5.3400879, -3.8939795, -1.2713313, 1.2928865

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656564
time: 5.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656550
time: 9.43 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.0376282, -4.4025211, -7.0503387, -4.3795691, -1.8686838, 1.8660555
1: -9.1213665, -6.9812369, -9.1307745, -6.9772372, -1.7135382, 1.7185845
2: -7.7147174, -5.9715805, -7.7209978, -5.9688101, -1.4813194, 1.4838645
3: -5.6720886, -3.6229084, -5.6883793, -3.6147442, -1.9623699, 1.9651132
4: -9.2664213, -7.2131119, -9.2687140, -7.2110744, -1.6923556, 1.7005477
5: 1.3468376, 2.7104490, 1.3402795, 2.7187390, -1.2042174, 1.2033582
6: -1.6796165, 0.3799267, -1.6833820, 0.3850634, -1.4393706, 1.4372740
7: -10.3810978, -8.7472286, -10.3848209, -8.7405357, -1.3546247, 1.3581216
8: 5.5748825, 7.2580900, 5.5665064, 7.2632279, -1.6883454, 1.6915836
9: -5.3514371, -3.8589723, -5.3557916, -3.8576891, -1.3045988, 1.3099842

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656570
time: 4.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656567
time: 4.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 32.08 seconds
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 32.08
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656564
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 32.08
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656550
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 32.08
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656570
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 32.08
Output dim: 8, lower bound: -0.8603363, upper bound: 0.8656567

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.0373912, -4.4025269, -7.0014420, -4.3937588, -1.8546753, 1.8028736
1: -9.1213589, -6.9812555, -9.1179619, -6.9893045, -1.7028561, 1.7060547
2: -7.7146020, -5.9715862, -7.6788635, -5.9890509, -1.4588881, 1.4465194
3: -5.6720743, -3.6230185, -5.6755323, -3.6492858, -1.9341536, 1.9469800
4: -9.2664137, -7.2132049, -9.2403831, -7.2537408, -1.6710129, 1.6696191
5: 1.3468416, 2.7103109, 1.3500416, 2.6916153, -1.1575217, 1.1837626
6: -1.6793177, 0.3799171, -1.6111054, 0.3605766, -1.4146185, 1.3787680
7: -10.3810759, -8.7473221, -10.3578730, -8.7680664, -1.3233886, 1.3293719
8: 5.5749550, 7.2580814, 5.5999775, 7.2382159, -1.6632609, 1.6581039
9: -5.3514280, -3.8591099, -5.3330078, -3.8957763, -1.2736430, 1.2860432

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 527

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656072
time: 4.45 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656547
time: 4.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.0373912, -4.4025269, -7.0024629, -4.3908386, -1.8572826, 1.8036313
1: -9.1213589, -6.9812555, -9.1236172, -6.9862590, -1.7063389, 1.7115135
2: -7.7146020, -5.9715862, -7.6828279, -5.9806948, -1.4674997, 1.4504318
3: -5.6720743, -3.6230185, -5.6804609, -3.6361613, -1.9431763, 1.9579029
4: -9.2664137, -7.2132049, -9.2557907, -7.2503576, -1.6654015, 1.6795602
5: 1.3468416, 2.7103109, 1.3473240, 2.7010965, -1.1661253, 1.1915164
6: -1.6793177, 0.3799171, -1.6131907, 0.3668208, -1.4196596, 1.3781104
7: -10.3810759, -8.7473221, -10.3694067, -8.7651787, -1.3208637, 1.3377900
8: 5.5749550, 7.2580814, 5.5951529, 7.2538633, -1.6789083, 1.6629286
9: -5.3514280, -3.8591099, -5.3400879, -3.8939812, -1.2713318, 1.2928863

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 527

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597325, upper bound: 0.8656076
time: 13.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656540
time: 6.13 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.0376282, -4.4025211, -7.0493312, -4.3824883, -1.8658543, 1.8640742
1: -9.1213665, -6.9812369, -9.1251087, -6.9802723, -1.7103863, 1.7130933
2: -7.7147174, -5.9715805, -7.7170486, -5.9771667, -1.4727092, 1.4741869
3: -5.6720886, -3.6229084, -5.6834531, -3.6278865, -1.9493532, 1.9537826
4: -9.2664213, -7.2131119, -9.2533016, -7.2144547, -1.6852884, 1.6847930
5: 1.3468376, 2.7104490, 1.3430372, 2.7092693, -1.1945753, 1.1938639
6: -1.6796165, 0.3799267, -1.6813068, 0.3788209, -1.4331002, 1.4333844
7: -10.3810978, -8.7472286, -10.3733025, -8.7434196, -1.3482714, 1.3463225
8: 5.5748825, 7.2580900, 5.5713263, 7.2475786, -1.6726961, 1.6867638
9: -5.3514371, -3.8589723, -5.3487177, -3.8594794, -1.3020744, 1.3031526

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 527

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656087
time: 5.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656553
time: 4.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.0376282, -4.4025211, -7.0503383, -4.3795710, -1.8695679, 1.8660560
1: -9.1213665, -6.9812369, -9.1307735, -6.9772377, -1.7138634, 1.7185841
2: -7.7147174, -5.9715805, -7.7209973, -5.9688115, -1.4813199, 1.4837780
3: -5.6720886, -3.6229084, -5.6883802, -3.6147449, -1.9583969, 1.9647789
4: -9.2664213, -7.2131119, -9.2687082, -7.2110720, -1.6920161, 1.7005472
5: 1.3468376, 2.7104490, 1.3402796, 2.7187386, -1.2042179, 1.2029176
6: -1.6796165, 0.3799267, -1.6833816, 0.3850627, -1.4393706, 1.4371166
7: -10.3810978, -8.7472286, -10.3848162, -8.7405357, -1.3543377, 1.3581204
8: 5.5748825, 7.2580900, 5.5665073, 7.2632256, -1.6883430, 1.6915827
9: -5.3514371, -3.8589723, -5.3557892, -3.8576884, -1.3045626, 1.3099847

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 527

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656077
time: 6.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656541
time: 7.02 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 34.56 seconds
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656072
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656547
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8597325, upper bound: 0.8656076
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656540
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656087
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656553
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8597315, upper bound: 0.8656077
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 34.56
Output dim: 8, lower bound: -0.8603348, upper bound: 0.8656541

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0364294, -4.4047379, -7.0012531, -4.3941774, -1.8036718, 1.8002000
1: -9.1178379, -6.9819107, -9.1172981, -6.9894304, -1.6968136, 1.7056389
2: -7.7127695, -5.9721537, -7.6785131, -5.9891572, -1.4556050, 1.4438989
3: -5.6712279, -3.6304796, -5.6753740, -3.6507025, -1.9298410, 1.9288940
4: -9.2625818, -7.2139940, -9.2396555, -7.2538958, -1.6653805, 1.6490912
5: 1.3480797, 2.7070789, 1.3502836, 2.6910033, -1.1540303, 1.1801291
6: -1.6784441, 0.3779655, -1.6109347, 0.3602061, -1.3850770, 1.3765697
7: -10.3775854, -8.7480316, -10.3572111, -8.7682056, -1.3196425, 1.3381069
8: 5.5771012, 7.2574873, 5.6003852, 7.2381010, -1.6609998, 1.6571021
9: -5.3478966, -3.8597322, -5.3323331, -3.8958945, -1.2705722, 1.2832673

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 527

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597323, upper bound: 0.8650504
time: 4.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597321, upper bound: 0.8656070
time: 4.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0433693, -4.4007416, -7.0014420, -4.3937597, -1.8575306, 1.8041124
1: -9.1244421, -6.9720364, -9.1179581, -6.9893060, -1.7130947, 1.7122965
2: -7.7164793, -5.9654446, -7.6788611, -5.9890518, -1.4604940, 1.4479556
3: -5.6968112, -3.6199942, -5.6755323, -3.6492953, -1.9456363, 1.9491806
4: -9.2691965, -7.2016706, -9.2403774, -7.2537417, -1.6755447, 1.6701770
5: 1.3356249, 2.7114618, 1.3500438, 2.6916101, -1.1697855, 1.1845903
6: -1.6851051, 0.3812337, -1.6111042, 0.3605726, -1.4171076, 1.3800292
7: -10.3838787, -8.7358780, -10.3578701, -8.7680664, -1.3265243, 1.3319104
8: 5.5716534, 7.2648034, 5.5999813, 7.2382154, -1.6665621, 1.6648221
9: -5.3533554, -3.8494527, -5.3330059, -3.8957760, -1.2770329, 1.2865162

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8597297, upper bound: 0.8653642
time: 4.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8603304, upper bound: 0.8656487
time: 4.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0364294, -4.4047379, -7.0022812, -4.3912621, -1.8064690, 1.8009596
1: -9.1178379, -6.9819107, -9.1229496, -6.9863844, -1.7002993, 1.7109847
2: -7.7127695, -5.9721537, -7.6824794, -5.9808011, -1.4642181, 1.4479656
3: -5.6712279, -3.6304796, -5.6803007, -3.6375787, -1.9388618, 1.9400768
4: -9.2625818, -7.2139940, -9.2550640, -7.2505112, -1.6597691, 1.6592751
5: 1.3480797, 2.7070789, 1.3475666, 2.7004871, -1.1621251, 1.1878805
6: -1.6784441, 0.3779655, -1.6130207, 0.3664494, -1.3913770, 1.3759112
7: -10.3775854, -8.7480316, -10.3687449, -8.7653160, -1.3171172, 1.3464563
8: 5.5771012, 7.2574873, 5.5955620, 7.2537489, -1.6766477, 1.6619253
9: -5.3478966, -3.8597322, -5.3394098, -3.8940983, -1.2683930, 1.2901108

Time for backsubstitution: 21.13 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.52 + 563.72 = 619.24 seconds
