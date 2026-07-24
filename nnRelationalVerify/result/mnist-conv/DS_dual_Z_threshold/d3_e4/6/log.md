## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1823463684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5293298, 2.5293295)
1: (-10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2577815, 2.2577815)
2: (-6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3718305, 2.3718295)
3: (-2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8441834, 1.8441832)
4: (-6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1593237, 3.1593237)
5: (-8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4321971, 2.4321966)
6: (-19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1931105, 3.1931105)
7: (4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772)
8: (-7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3909245, 2.3909245)
9: (-7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6847959, 2.6847959)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.98 + 34.86 = 57.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.1847158, upper bound: 1.1847152

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837863, upper bound: 1.1847078
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847085, upper bound: 1.1837853
time: 4.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.65 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.65
Output dim: 7, lower bound: -1.1837863, upper bound: 1.1847078
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.65
Output dim: 7, lower bound: -1.1847085, upper bound: 1.1837853

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5265794, 2.5296471
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2568817, 2.2578845
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3719592, 2.3707409
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8445597, 1.8409653
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1582098, 3.1594534
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4309306, 2.4323487
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1932402, 3.1920815
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3911119, 2.3893216
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6825171, 2.6850801

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6209

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837851, upper bound: 1.1827209
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1817996, upper bound: 1.1847066
time: 5.40 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5293298, 2.5265791
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2577815, 2.2568812
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3707414, 2.3718295
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8409653, 1.8441832
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1593237, 3.1582098
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4321971, 2.4309301
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1920815, 3.1931105
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3893218, 2.3909245
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6847959, 2.6825166

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6209

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847074, upper bound: 1.1817985
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827218, upper bound: 1.1837845
time: 4.69 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.57 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 7, lower bound: -1.1837851, upper bound: 1.1827209
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 7, lower bound: -1.1817996, upper bound: 1.1847066
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 7, lower bound: -1.1847074, upper bound: 1.1817985
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 7, lower bound: -1.1827218, upper bound: 1.1837845

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4773202, 2.4733636
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2305064, 2.2277374
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3125024, 2.3189664
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8105974, 1.8114324
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1410475, 3.1395650
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4087391, 2.4140487
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1643705, 3.1590986
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3881531, 2.3848557
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6601686, 2.6588163

Time for backsubstitution: 21.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837792, upper bound: 1.1782352
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1793017, upper bound: 1.1827153
time: 4.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4702954, 2.4803882
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2267337, 2.2315102
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3201847, 2.3112836
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8150272, 1.8070028
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1383219, 3.1422901
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4126301, 2.4101577
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1602573, 3.1632118
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3866458, 2.3863628
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6562529, 2.6627316

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1817937, upper bound: 1.1802232
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1773161, upper bound: 1.1847009
time: 4.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4800706, 2.4702959
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2314067, 2.2267342
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3112836, 2.3200564
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8070030, 1.8146505
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1421614, 3.1383214
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4100056, 2.4126301
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1632118, 3.1601267
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3863626, 2.3864594
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6624475, 2.6562529

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847015, upper bound: 1.1773151
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1802240, upper bound: 1.1817925
time: 4.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4730458, 2.4773204
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2276330, 2.2305069
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3189664, 2.3123736
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8114328, 1.8102210
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1394358, 3.1410470
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4138966, 2.4087391
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1590986, 3.1642408
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3848557, 2.3879664
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6585321, 2.6601686

Time for backsubstitution: 21.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827160, upper bound: 1.1793012
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1782359, upper bound: 1.1837782
time: 5.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1837792, upper bound: 1.1782352
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1793017, upper bound: 1.1827153
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1817937, upper bound: 1.1802232
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1773161, upper bound: 1.1847009
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1847015, upper bound: 1.1773151
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1802240, upper bound: 1.1817925
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1827160, upper bound: 1.1793012
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.63
Output dim: 7, lower bound: -1.1782359, upper bound: 1.1837782

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4761701, 2.4727309
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2284951, 2.2240810
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3062277, 2.3155184
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8093462, 1.8107433
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1328030, 3.1245766
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4086494, 2.4140034
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1621428, 3.1578770
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3843737, 2.3827775
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6591973, 2.6570463

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837771, upper bound: 1.1748424
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1803970, upper bound: 1.1782331
time: 4.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4766879, 2.4722128
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2268510, 2.2257252
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3090544, 2.3126917
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8099079, 1.8101811
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1260586, 3.1313210
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4086933, 2.4139581
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1631498, 3.1568699
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3860750, 2.3810761
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6583991, 2.6578450

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1792995, upper bound: 1.1793263
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1759195, upper bound: 1.1827130
time: 4.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4696631, 2.4792373
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2230783, 2.2294989
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3167372, 2.3050089
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8143377, 1.8057516
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1233330, 3.1340461
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4125843, 2.4100671
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1590357, 3.1609831
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3845677, 2.3825831
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6544838, 2.6617603

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1773141, upper bound: 1.1813185
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1739293, upper bound: 1.1846990
time: 4.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4789200, 2.4696629
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2293944, 2.2230778
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3050089, 2.3166089
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8057518, 1.8139613
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1339178, 3.1233330
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4099150, 2.4125853
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1609840, 3.1589060
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3825836, 2.3843808
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6614761, 2.6544833

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846993, upper bound: 1.1739267
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1813193, upper bound: 1.1773132
time: 4.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4718952, 2.4766874
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2256207, 2.2268515
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3126917, 2.3089261
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8101816, 1.8095315
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1311922, 3.1260586
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4138060, 2.4086943
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1568699, 3.1630201
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3810763, 2.3858879
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6575608, 2.6583986

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827139, upper bound: 1.1759189
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793271, upper bound: 1.1792988
time: 5.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4724131, 2.4761693
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2239766, 2.2284951
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3155184, 2.3060989
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8107433, 1.8089695
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1244478, 3.1328030
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4138517, 2.4086485
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1578770, 3.1620131
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3827777, 2.3841865
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6567626, 2.6591973

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1782339, upper bound: 1.1803963
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1748431, upper bound: 1.1837759
time: 5.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1837771, upper bound: 1.1748424
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1803970, upper bound: 1.1782331
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1792995, upper bound: 1.1793263
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1759195, upper bound: 1.1827130
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1773141, upper bound: 1.1813185
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1739293, upper bound: 1.1846990
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1846993, upper bound: 1.1739267
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1813193, upper bound: 1.1773132
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1827139, upper bound: 1.1759189
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1793271, upper bound: 1.1792988
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1782339, upper bound: 1.1803963
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.95
Output dim: 7, lower bound: -1.1748431, upper bound: 1.1837759

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4664264, 2.4599323
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2310748, 2.2263484
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2901688, 2.3014627
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7607675, 1.7682376
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1251497, 3.1158338
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3583026, 2.3699479
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1405354, 3.1331854
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3632503, 2.3577976
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6539125, 2.6510067

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1816301, upper bound: 1.1748414
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837760, upper bound: 1.1727015
time: 4.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4638886, 2.4624689
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2291179, 2.2283044
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2949967, 2.2966328
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7674017, 1.7616029
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1173162, 3.1236677
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3646388, 2.3636112
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1384583, 3.1352606
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3610954, 2.3599524
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6523590, 2.6525593

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1737726, upper bound: 1.1827120
time: 5.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1759184, upper bound: 1.1805658
time: 5.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4568639, 2.4694946
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2253451, 2.2320781
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3026810, 2.2889500
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7718315, 1.7571733
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1145906, 3.1263933
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3685298, 2.3597202
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1343451, 3.1393766
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3595881, 2.3614597
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6484437, 2.6564755

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1717804, upper bound: 1.1846973
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1739262, upper bound: 1.1825519
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4691768, 2.4568644
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2319741, 2.2253451
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2889500, 2.3025522
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7571735, 1.7714555
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1262655, 3.1145902
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3595691, 2.3685293
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1393766, 3.1342134
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3614602, 2.3594017
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6561913, 2.6484437

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825524, upper bound: 1.1739254
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1717794
time: 5.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4621511, 2.4638889
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2282004, 2.2291183
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2966328, 2.2948680
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7616029, 1.7670257
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1235399, 3.1173158
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3634601, 2.3646388
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1352606, 3.1383276
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3599529, 2.3609087
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6522751, 2.6523590

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1805670, upper bound: 1.1759177
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827128, upper bound: 1.1737715
time: 5.48 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1816301, upper bound: 1.1748414
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1837760, upper bound: 1.1727015
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1737726, upper bound: 1.1827120
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1759184, upper bound: 1.1805658
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1717804, upper bound: 1.1846973
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1739262, upper bound: 1.1825519
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1825524, upper bound: 1.1739254
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1846982, upper bound: 1.1717794
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1805670, upper bound: 1.1759177
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.12
Output dim: 7, lower bound: -1.1827128, upper bound: 1.1737715
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.12
Output dim: 7, lower bound: -1.1748431, upper bound: 1.1837759

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.84 + 544.60 = 602.44 seconds
