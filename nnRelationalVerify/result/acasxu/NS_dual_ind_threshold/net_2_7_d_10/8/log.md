## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 20.6039733455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-121.3309937, 166.6516571, -121.3309937, 166.6516571, -287.9826355, 287.9826050)
1: (-15.4075546, 14.3914642, -15.4075546, 14.3914642, -29.7990189, 29.7990170)
2: (-8.8996906, 14.6720200, -8.8996906, 14.6720200, -23.5717106, 23.5717106)
3: (-7.2209902, 16.0608215, -7.2209902, 16.0608215, -23.2818108, 23.2818108)
4: (-10.9869347, 13.1586323, -10.9869347, 13.1586323, -24.1455631, 24.1455631)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.69 + 1.66 = 4.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -20.7075109, upper bound: 20.7075081

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284488, upper bound: 20.7068638
time: 0.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284393
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.34 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 3, lower bound: -20.6284488, upper bound: 20.7068638
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284393

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -112.6360321, 155.6537476, -121.3309937, 166.6516571, -279.2876587, 276.9847412
1: -14.4320812, 13.3855782, -15.4075546, 14.3914642, -28.8235455, 28.7931328
2: -8.2786350, 13.6952105, -8.8996906, 14.6720200, -22.9506493, 22.5949020
3: -6.6903615, 15.0224915, -7.2209902, 16.0608215, -22.7511826, 22.2434807
4: -10.2362566, 12.2680149, -10.9869347, 13.1586323, -23.3948860, 23.2549477

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284361
time: 0.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284382
time: 0.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -134.5602264, 187.2960663, -121.3309937, 166.6516571, -301.2118835, 308.6270752
1: -17.3629227, 16.0680561, -15.4075546, 14.3914642, -31.7543850, 31.4756107
2: -9.9258614, 16.4582443, -8.8996906, 14.6720200, -24.5978813, 25.3579350
3: -8.0332489, 18.0502758, -7.2209902, 16.0608215, -24.0940685, 25.2712669
4: -12.2632847, 14.7442722, -10.9869347, 13.1586323, -25.4219170, 25.7312069

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284393
time: 0.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284369
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.85 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284361
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284382
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284393
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 3, lower bound: -20.6284398, upper bound: 20.6284369

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -112.6360321, 155.6537476, -112.6360321, 155.6537476, -268.2897949, 268.2897949
1: -14.4320812, 13.3855782, -14.4320812, 13.3855782, -27.8176594, 27.8176594
2: -8.2786350, 13.6952105, -8.2786350, 13.6952105, -21.9738388, 21.9738388
3: -6.6903615, 15.0224915, -6.6903615, 15.0224915, -21.7128525, 21.7128525
4: -10.2362566, 12.2680149, -10.2362566, 12.2680149, -22.5042725, 22.5042725

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096325, upper bound: 20.7031024
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096329, upper bound: 20.6938723
time: 0.55 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -112.6360321, 155.6537476, -134.5602264, 187.2960663, -299.9320984, 290.2139893
1: -14.4320812, 13.3855782, -17.3629227, 16.0680561, -30.5001373, 30.7485008
2: -8.2786350, 13.6952105, -9.9258614, 16.4582443, -24.7368755, 23.6210709
3: -6.6903615, 15.0224915, -8.0332489, 18.0502758, -24.7406368, 23.0557404
4: -10.2362566, 12.2680149, -12.2632847, 14.7442722, -24.9805298, 24.5312996

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096325, upper bound: 20.7030987
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096329, upper bound: 20.6938718
time: 0.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -134.5602264, 187.2960663, -112.6360321, 155.6537476, -290.2139893, 299.9320984
1: -17.3629227, 16.0680561, -14.4320812, 13.3855782, -30.7485008, 30.5001354
2: -9.9258614, 16.4582443, -8.2786350, 13.6952105, -23.6210709, 24.7368755
3: -8.0332489, 18.0502758, -6.6903615, 15.0224915, -23.0557404, 24.7406368
4: -12.2632847, 14.7442722, -10.2362566, 12.2680149, -24.5312996, 24.9805298

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096234, upper bound: 20.6202055
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096224
time: 0.51 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -134.5602264, 187.2960663, -134.5602264, 187.2960663, -321.8562927, 321.8562927
1: -17.3629227, 16.0680561, -17.3629227, 16.0680561, -33.4309692, 33.4309692
2: -9.9258614, 16.4582443, -9.9258614, 16.4582443, -26.3841057, 26.3841057
3: -8.0332489, 18.0502758, -8.0332489, 18.0502758, -26.0835247, 26.0835247
4: -12.2632847, 14.7442722, -12.2632847, 14.7442722, -27.0075569, 27.0075569

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096234, upper bound: 20.6202070
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096221
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096325, upper bound: 20.7031024
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096329, upper bound: 20.6938723
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096325, upper bound: 20.7030987
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096329, upper bound: 20.6938718
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096234, upper bound: 20.6202055
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096224
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096234, upper bound: 20.6202070
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.86
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096221

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -112.6360321, 155.6537476, -257.2683105, 253.4813080
1: -13.0697832, 12.1037197, -14.4320812, 13.3855782, -26.4553585, 26.5358009
2: -7.4919033, 12.3726549, -8.2786350, 13.6952105, -21.1871052, 20.6512890
3: -6.0223894, 13.5765505, -6.6903615, 15.0224915, -21.0448799, 20.2669106
4: -9.2628574, 11.0729675, -10.2362566, 12.2680149, -21.5308723, 21.3092232

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938857, upper bound: 20.6938849
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938857, upper bound: 20.6938807
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -112.6360321, 155.6537476, -256.2481079, 252.3153839
1: -12.9418774, 12.0215712, -14.4320812, 13.3855782, -26.3274555, 26.4536514
2: -7.4400887, 12.2674332, -8.2786350, 13.6952105, -21.1352978, 20.5460663
3: -6.0007830, 13.4309692, -6.6903615, 15.0224915, -21.0232735, 20.1213303
4: -9.1817017, 10.9723797, -10.2362566, 12.2680149, -21.4497166, 21.2086372

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938861, upper bound: 20.6938836
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938861, upper bound: 20.6938844
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -134.5602264, 187.2960663, -288.9106445, 275.4055176
1: -13.0697832, 12.1037197, -17.3629227, 16.0680561, -29.1378326, 29.4666386
2: -7.4919033, 12.3726549, -9.9258614, 16.4582443, -23.9501438, 22.2985153
3: -6.0223894, 13.5765505, -8.0332489, 18.0502758, -24.0726662, 21.6097946
4: -9.2628574, 11.0729675, -12.2632847, 14.7442722, -24.0071297, 23.3362522

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096325, upper bound: 20.6938733
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096325, upper bound: 20.6938752
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -134.5602264, 187.2960663, -287.8904114, 274.2395630
1: -12.9418774, 12.0215712, -17.3629227, 16.0680561, -29.0099316, 29.3844891
2: -7.4400887, 12.2674332, -9.9258614, 16.4582443, -23.8983326, 22.1932945
3: -6.0007830, 13.4309692, -8.0332489, 18.0502758, -24.0510597, 21.4642162
4: -9.1817017, 10.9723797, -12.2632847, 14.7442722, -23.9259739, 23.2356644

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096330, upper bound: 20.6938751
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096330, upper bound: 20.6938720
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -112.6360321, 155.6537476, -278.7831726, 284.7149963
1: -15.9661789, 14.7359285, -14.4320812, 13.3855782, -29.3517570, 29.1680107
2: -9.1003685, 15.1034212, -8.2786350, 13.6952105, -22.7955761, 23.3820534
3: -7.3377576, 16.5662975, -6.6903615, 15.0224915, -22.3602486, 23.2566586
4: -11.2568188, 13.5198822, -10.2362566, 12.2680149, -23.5248337, 23.7561359

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938761, upper bound: 20.6096323
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938761, upper bound: 20.6096281
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -112.6360321, 155.6537476, -277.3876953, 283.0082397
1: -15.7912579, 14.6004047, -14.4320812, 13.3855782, -29.1768360, 29.0324841
2: -9.0307693, 14.9529743, -8.2786350, 13.6952105, -22.7259789, 23.2316093
3: -7.2881451, 16.3678722, -6.6903615, 15.0224915, -22.3106365, 23.0582333
4: -11.1406412, 13.3778372, -10.2362566, 12.2680149, -23.4086571, 23.6140938

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938763, upper bound: 20.6096301
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938763, upper bound: 20.6096325
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -134.5602264, 187.2960663, -310.4254761, 306.6391907
1: -15.9661789, 14.7359285, -17.3629227, 16.0680561, -32.0342255, 32.0988503
2: -9.1003685, 15.1034212, -9.9258614, 16.4582443, -25.5586109, 25.0292816
3: -7.3377576, 16.5662975, -8.0332489, 18.0502758, -25.3880329, 24.5995445
4: -11.2568188, 13.5198822, -12.2632847, 14.7442722, -26.0010910, 25.7831669

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096230, upper bound: 20.6096215
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096230, upper bound: 20.6096208
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -134.5602264, 187.2960663, -309.0299683, 304.9324341
1: -15.7912579, 14.6004047, -17.3629227, 16.0680561, -31.8593063, 31.9633236
2: -9.0307693, 14.9529743, -9.9258614, 16.4582443, -25.4890137, 24.8788357
3: -7.2881451, 16.3678722, -8.0332489, 18.0502758, -25.3384209, 24.4011211
4: -11.1406412, 13.3778372, -12.2632847, 14.7442722, -25.8849144, 25.6411209

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096201
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096199
time: 0.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.94 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938857, upper bound: 20.6938849
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938857, upper bound: 20.6938807
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938861, upper bound: 20.6938836
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938861, upper bound: 20.6938844
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096325, upper bound: 20.6938733
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096325, upper bound: 20.6938752
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096330, upper bound: 20.6938751
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096330, upper bound: 20.6938720
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938761, upper bound: 20.6096323
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938761, upper bound: 20.6096281
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938763, upper bound: 20.6096301
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6938763, upper bound: 20.6096325
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096230, upper bound: 20.6096215
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096230, upper bound: 20.6096208
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096201
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 3, lower bound: -20.6096232, upper bound: 20.6096199

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -101.6145782, 140.8452759, -242.4598389, 242.4598541
1: -13.0697832, 12.1037197, -13.0697832, 12.1037197, -25.1735001, 25.1735001
2: -7.4919033, 12.3726549, -7.4919033, 12.3726549, -19.8645554, 19.8645554
3: -6.0223894, 13.5765505, -6.0223894, 13.5765505, -19.5989380, 19.5989361
4: -9.2628574, 11.0729675, -9.2628574, 11.0729675, -20.3358250, 20.3358250

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6929287, upper bound: 20.6962296
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6929301, upper bound: 20.7017038
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -100.5943604, 139.6793518, -241.2939301, 241.4396362
1: -13.0697832, 12.1037197, -12.9418774, 12.0215712, -25.0913506, 25.0455971
2: -7.4919033, 12.3726549, -7.4400887, 12.2674332, -19.7593307, 19.8127441
3: -6.0223894, 13.5765505, -6.0007830, 13.4309692, -19.4533577, 19.5773296
4: -9.2628574, 11.0729675, -9.1817017, 10.9723797, -20.2352371, 20.2546692

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6929287, upper bound: 20.6962297
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6929301, upper bound: 20.7017048
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -101.6145782, 140.8452759, -241.4396362, 241.2939301
1: -12.9418774, 12.0215712, -13.0697832, 12.1037197, -25.0455971, 25.0913506
2: -7.4400887, 12.2674332, -7.4919033, 12.3726549, -19.8127441, 19.7593307
3: -6.0007830, 13.4309692, -6.0223894, 13.5765505, -19.5773296, 19.4533577
4: -9.1817017, 10.9723797, -9.2628574, 11.0729675, -20.2546692, 20.2352371

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824028, upper bound: 20.6917045
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824005
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -100.5943604, 139.6793518, -240.2737122, 240.2737122
1: -12.9418774, 12.0215712, -12.9418774, 12.0215712, -24.9634476, 24.9634476
2: -7.4400887, 12.2674332, -7.4400887, 12.2674332, -19.7075214, 19.7075214
3: -6.0007830, 13.4309692, -6.0007830, 13.4309692, -19.4317513, 19.4317513
4: -9.1817017, 10.9723797, -9.1817017, 10.9723797, -20.1540794, 20.1540794

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824028, upper bound: 20.6917046
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824010, upper bound: 20.6823966
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -123.1294250, 172.0789642, -273.6935425, 263.9747009
1: -13.0697832, 12.1037197, -15.9661789, 14.7359285, -27.8057117, 28.0698929
2: -7.4919033, 12.3726549, -9.1003685, 15.1034212, -22.5953197, 21.4730225
3: -6.0223894, 13.5765505, -7.3377576, 16.5662975, -22.5886879, 20.9143047
4: -9.2628574, 11.0729675, -11.2568188, 13.5198822, -22.7827377, 22.3297863

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6076449, upper bound: 20.6962190
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6076462, upper bound: 20.7016898
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -101.6145782, 140.8452759, -121.7339630, 170.3722076, -271.9867859, 262.5792236
1: -13.0697832, 12.1037197, -15.7912579, 14.6004047, -27.6701832, 27.8949757
2: -7.4919033, 12.3726549, -9.0307693, 14.9529743, -22.4448738, 21.4034233
3: -6.0223894, 13.5765505, -7.2881451, 16.3678722, -22.3902626, 20.8646889
4: -9.2628574, 11.0729675, -11.1406412, 13.3778372, -22.6406937, 22.2136078

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6076449, upper bound: 20.6962174
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6076462, upper bound: 20.7016941
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -123.1294250, 172.0789642, -272.6733093, 262.8087769
1: -12.9418774, 12.0215712, -15.9661789, 14.7359285, -27.6778069, 27.9877472
2: -7.4400887, 12.2674332, -9.1003685, 15.1034212, -22.5435104, 21.3678017
3: -6.0007830, 13.4309692, -7.3377576, 16.5662975, -22.5670776, 20.7687263
4: -9.1817017, 10.9723797, -11.2568188, 13.5198822, -22.7015781, 22.2291985

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839521, upper bound: 20.6916923
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823841
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -100.5943604, 139.6793518, -121.7339630, 170.3722076, -270.9665222, 261.4133301
1: -12.9418774, 12.0215712, -15.7912579, 14.6004047, -27.5422821, 27.8128262
2: -7.4400887, 12.2674332, -9.0307693, 14.9529743, -22.3930626, 21.2982025
3: -6.0007830, 13.4309692, -7.2881451, 16.3678722, -22.3686543, 20.7191124
4: -9.1817017, 10.9723797, -11.1406412, 13.3778372, -22.5595379, 22.1130219

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839521, upper bound: 20.6916923
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823886
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -101.6145782, 140.8452759, -263.9747009, 273.6935425
1: -15.9661789, 14.7359285, -13.0697832, 12.1037197, -28.0698986, 27.8057117
2: -9.1003685, 15.1034212, -7.4919033, 12.3726549, -21.4730225, 22.5953217
3: -7.3377576, 16.5662975, -6.0223894, 13.5765505, -20.9143047, 22.5886879
4: -11.2568188, 13.5198822, -9.2628574, 11.0729675, -22.3297863, 22.7827377

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938428, upper bound: 20.5632867
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938448, upper bound: 20.6065727
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -100.5943604, 139.6793518, -262.8087769, 272.6733093
1: -15.9661789, 14.7359285, -12.9418774, 12.0215712, -27.9877491, 27.6778069
2: -9.1003685, 15.1034212, -7.4400887, 12.2674332, -21.3678017, 22.5435104
3: -7.3377576, 16.5662975, -6.0007830, 13.4309692, -20.7687263, 22.5670795
4: -11.2568188, 13.5198822, -9.1817017, 10.9723797, -22.2291985, 22.7015781

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938430, upper bound: 20.5632878
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938475, upper bound: 20.6065708
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -101.6145782, 140.8452759, -262.5792236, 271.9867554
1: -15.7912579, 14.6004047, -13.0697832, 12.1037197, -27.8949738, 27.6701851
2: -9.0307693, 14.9529743, -7.4919033, 12.3726549, -21.4034233, 22.4448738
3: -7.2881451, 16.3678722, -6.0223894, 13.5765505, -20.8646889, 22.3902626
4: -11.1406412, 13.3778372, -9.2628574, 11.0729675, -22.2136078, 22.6406937

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938424, upper bound: 20.5481536
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938447, upper bound: 20.5957442
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -100.5943604, 139.6793518, -261.4133301, 270.9665222
1: -15.7912579, 14.6004047, -12.9418774, 12.0215712, -27.8128281, 27.5422821
2: -9.0307693, 14.9529743, -7.4400887, 12.2674332, -21.2982025, 22.3930626
3: -7.2881451, 16.3678722, -6.0007830, 13.4309692, -20.7191124, 22.3686543
4: -11.1406412, 13.3778372, -9.1817017, 10.9723797, -22.1130219, 22.5595379

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938424, upper bound: 20.5481519
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938447, upper bound: 20.5957428
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -123.1294250, 172.0789642, -295.2083740, 295.2083740
1: -15.9661789, 14.7359285, -15.9661789, 14.7359285, -30.7021065, 30.7021065
2: -9.1003685, 15.1034212, -9.1003685, 15.1034212, -24.2037888, 24.2037888
3: -7.3377576, 16.5662975, -7.3377576, 16.5662975, -23.9040546, 23.9040527
4: -11.2568188, 13.5198822, -11.2568188, 13.5198822, -24.7766991, 24.7766991

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957279, upper bound: 20.5632733
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5957304, upper bound: 20.6065585
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -123.1294250, 172.0789642, -121.7339630, 170.3722076, -293.5015869, 293.8128662
1: -15.9661789, 14.7359285, -15.7912579, 14.6004047, -30.5665817, 30.5271873
2: -9.1003685, 15.1034212, -9.0307693, 14.9529743, -24.0533428, 24.1341896
3: -7.3377576, 16.5662975, -7.2881451, 16.3678722, -23.7056293, 23.8544426
4: -11.2568188, 13.5198822, -11.1406412, 13.3778372, -24.6346550, 24.6605225

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957282, upper bound: 20.5632679
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5957304, upper bound: 20.6065577
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -123.1294250, 172.0789642, -293.8128662, 293.5015869
1: -15.7912579, 14.6004047, -15.9661789, 14.7359285, -30.5271873, 30.5665741
2: -9.0307693, 14.9529743, -9.1003685, 15.1034212, -24.1341896, 24.0533409
3: -7.2881451, 16.3678722, -7.3377576, 16.5662975, -23.8544388, 23.7056293
4: -11.1406412, 13.3778372, -11.2568188, 13.5198822, -24.6605225, 24.6346550

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957246, upper bound: 20.5481427
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957320, upper bound: 20.5957308
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -121.7339630, 170.3722076, -121.7339630, 170.3722076, -292.1060791, 292.1060791
1: -15.7912579, 14.6004047, -15.7912579, 14.6004047, -30.3916569, 30.3916588
2: -9.0307693, 14.9529743, -9.0307693, 14.9529743, -23.9837437, 23.9837437
3: -7.2881451, 16.3678722, -7.2881451, 16.3678722, -23.6560154, 23.6560173
4: -11.1406412, 13.3778372, -11.1406412, 13.3778372, -24.5184784, 24.5184784

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957243, upper bound: 20.5481401
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5957320, upper bound: 20.5957284
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.04 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6929287, upper bound: 20.6962296
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6929301, upper bound: 20.7017038
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6929287, upper bound: 20.6962297
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6929301, upper bound: 20.7017048
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6824028, upper bound: 20.6917045
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824005
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6824028, upper bound: 20.6917046
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6824010, upper bound: 20.6823966
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6076449, upper bound: 20.6962190
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6076462, upper bound: 20.7016898
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6076449, upper bound: 20.6962174
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6076462, upper bound: 20.7016941
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5839521, upper bound: 20.6916923
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823841
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5839521, upper bound: 20.6916923
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823886
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938428, upper bound: 20.5632867
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938448, upper bound: 20.6065727
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938430, upper bound: 20.5632878
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938475, upper bound: 20.6065708
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938424, upper bound: 20.5481536
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938447, upper bound: 20.5957442
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938424, upper bound: 20.5481519
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.6938447, upper bound: 20.5957428
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957279, upper bound: 20.5632733
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957304, upper bound: 20.6065585
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957282, upper bound: 20.5632679
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957304, upper bound: 20.6065577
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957246, upper bound: 20.5481427
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957320, upper bound: 20.5957308
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957243, upper bound: 20.5481401
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 3, lower bound: -20.5957320, upper bound: 20.5957284

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -101.6145782, 140.8452759, -237.5531311, 236.7829437
1: -12.5404415, 11.6070614, -13.0697832, 12.1037197, -24.6441574, 24.6768417
2: -7.1486292, 11.8655405, -7.4919033, 12.3726549, -19.5212841, 19.3574409
3: -5.7478900, 13.0301399, -6.0223894, 13.5765505, -19.3244343, 19.0525284
4: -8.8640528, 10.6130219, -9.2628574, 11.0729675, -19.9370193, 19.8758793

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962286, upper bound: 20.6962276
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962286, upper bound: 20.6962297
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -101.6145782, 140.8452759, -235.9187927, 234.3688660
1: -12.3337259, 11.3693285, -13.0697832, 12.1037197, -24.4374447, 24.4391098
2: -7.0452828, 11.6368580, -7.4919033, 12.3726549, -19.4179382, 19.1287613
3: -5.6269884, 12.7925835, -6.0223894, 13.5765505, -19.2035370, 18.8149700
4: -8.6930714, 10.4093132, -9.2628574, 11.0729675, -19.7660389, 19.6721687

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962304, upper bound: 20.7017027
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962304, upper bound: 20.7017003
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -100.5943604, 139.6793518, -236.3872223, 235.7627106
1: -12.5404415, 11.6070614, -12.9418774, 12.0215712, -24.5620041, 24.5489388
2: -7.1486292, 11.8655405, -7.4400887, 12.2674332, -19.4160595, 19.3056297
3: -5.7478900, 13.0301399, -6.0007830, 13.4309692, -19.1788578, 19.0309200
4: -8.8640528, 10.6130219, -9.1817017, 10.9723797, -19.8364296, 19.7947235

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6907926, upper bound: 20.6938734
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804714, upper bound: 20.6938725
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -100.5943604, 139.6793518, -234.7528687, 233.3486481
1: -12.3337259, 11.3693285, -12.9418774, 12.0215712, -24.3552952, 24.3112068
2: -7.0452828, 11.6368580, -7.4400887, 12.2674332, -19.3127136, 19.0769463
3: -5.6269884, 12.7925835, -6.0007830, 13.4309692, -19.0579567, 18.7933617
4: -8.6930714, 10.4093132, -9.1817017, 10.9723797, -19.6654510, 19.5910130

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6907933, upper bound: 20.6987052
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804721, upper bound: 20.6987014
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -101.6145782, 140.8452759, -240.0036163, 239.2704620
1: -12.7440815, 11.8651171, -13.0697832, 12.1037197, -24.8478012, 24.9348946
2: -7.3402557, 12.0819092, -7.4919033, 12.3726549, -19.7129097, 19.5738106
3: -5.9315467, 13.2485046, -6.0223894, 13.5765505, -19.5080929, 19.2708931
4: -9.0659151, 10.8102217, -9.2628574, 11.0729675, -20.1388798, 20.0730782

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6992308, upper bound: 20.6716533
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716528
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -101.6145782, 140.8452759, -237.4929047, 236.0250397
1: -12.4564848, 11.5711174, -13.0697832, 12.1037197, -24.5601997, 24.6408958
2: -7.1575665, 11.7965927, -7.4919033, 12.3726549, -19.5302219, 19.2884903
3: -5.7723017, 12.9366655, -6.0223894, 13.5765505, -19.3488503, 18.9590549
4: -8.8456087, 10.5499878, -9.2628574, 11.0729675, -19.9185753, 19.8128452

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6992307, upper bound: 20.6708115
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708099
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -100.5943604, 139.6793518, -238.8376923, 238.2502289
1: -12.7440815, 11.8651171, -12.9418774, 12.0215712, -24.7656517, 24.8069954
2: -7.3402557, 12.0819092, -7.4400887, 12.2674332, -19.6076870, 19.5219975
3: -5.9315467, 13.2485046, -6.0007830, 13.4309692, -19.3625145, 19.2492867
4: -9.0659151, 10.8102217, -9.1817017, 10.9723797, -20.0382938, 19.9919224

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824014
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824008
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -100.5943604, 139.6793518, -236.3269806, 235.0048218
1: -12.4564848, 11.5711174, -12.9418774, 12.0215712, -24.4780540, 24.5129948
2: -7.1575665, 11.7965927, -7.4400887, 12.2674332, -19.4249973, 19.2366810
3: -5.7723017, 12.9366655, -6.0007830, 13.4309692, -19.2032700, 18.9374466
4: -8.8456087, 10.5499878, -9.1817017, 10.9723797, -19.8179855, 19.7316895

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6823972
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824011
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -123.1294250, 172.0789642, -268.7867432, 258.2977905
1: -12.5404415, 11.6070614, -15.9661789, 14.7359285, -27.2763710, 27.5732365
2: -7.1486292, 11.8655405, -9.1003685, 15.1034212, -22.2520485, 20.9659081
3: -5.7478900, 13.0301399, -7.3377576, 16.5662975, -22.3141861, 20.3678932
4: -8.8640528, 10.6130219, -11.2568188, 13.5198822, -22.3839283, 21.8698406

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6190990, upper bound: 20.6805913
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805858
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -123.1294250, 172.0789642, -267.1523743, 255.8837280
1: -12.3337259, 11.3693285, -15.9661789, 14.7359285, -27.0696545, 27.3355064
2: -7.0452828, 11.6368580, -9.1003685, 15.1034212, -22.1487026, 20.7372265
3: -5.6269884, 12.7925835, -7.3377576, 16.5662975, -22.1932869, 20.1303368
4: -8.6930714, 10.4093132, -11.2568188, 13.5198822, -22.2129498, 21.6661320

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6190992, upper bound: 20.6831516
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885817, upper bound: 20.6831457
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -121.7339630, 170.3722076, -267.0799866, 256.9023132
1: -12.5404415, 11.6070614, -15.7912579, 14.6004047, -27.1408443, 27.3983154
2: -7.1486292, 11.8655405, -9.0307693, 14.9529743, -22.1016045, 20.8963089
3: -5.7478900, 13.0301399, -7.2881451, 16.3678722, -22.1157627, 20.3182793
4: -8.8640528, 10.6130219, -11.1406412, 13.3778372, -22.2418880, 21.7536621

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6055122, upper bound: 20.6938608
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813300, upper bound: 20.6938580
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -121.7339630, 170.3722076, -265.4456177, 254.4882202
1: -12.3337259, 11.3693285, -15.7912579, 14.6004047, -26.9341316, 27.1605873
2: -7.0452828, 11.6368580, -9.0307693, 14.9529743, -21.9982567, 20.6676273
3: -5.6269884, 12.7925835, -7.2881451, 16.3678722, -21.9948616, 20.0807266
4: -8.6930714, 10.4093132, -11.1406412, 13.3778372, -22.0709076, 21.5499535

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6055129, upper bound: 20.6986913
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813307, upper bound: 20.6986907
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -123.1294250, 172.0789642, -271.2373047, 260.7853088
1: -12.7440815, 11.8651171, -15.9661789, 14.7359285, -27.4800110, 27.8312912
2: -7.3402557, 12.0819092, -9.1003685, 15.1034212, -22.4436760, 21.1822777
3: -5.9315467, 13.2485046, -7.3377576, 16.5662975, -22.4978428, 20.5862598
4: -9.0659151, 10.8102217, -11.2568188, 13.5198822, -22.5857906, 22.0670395

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6116696, upper bound: 20.6716446
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716384
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -123.1294250, 172.0789642, -268.7265930, 257.5398865
1: -12.4564848, 11.5711174, -15.9661789, 14.7359285, -27.1924133, 27.5372906
2: -7.1575665, 11.7965927, -9.1003685, 15.1034212, -22.2609882, 20.8969612
3: -5.7723017, 12.9366655, -7.3377576, 16.5662975, -22.3385983, 20.2744217
4: -8.8456087, 10.5499878, -11.2568188, 13.5198822, -22.3654842, 21.8068066

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6116695, upper bound: 20.6708002
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6708002
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -121.7339630, 170.3722076, -269.5305176, 259.3898010
1: -12.7440815, 11.8651171, -15.7912579, 14.6004047, -27.3444862, 27.6563759
2: -7.3402557, 12.0819092, -9.0307693, 14.9529743, -22.2932301, 21.1126785
3: -5.9315467, 13.2485046, -7.2881451, 16.3678722, -22.2994175, 20.5366440
4: -9.0659151, 10.8102217, -11.1406412, 13.3778372, -22.4437523, 21.9508629

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823880
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823855
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -121.7339630, 170.3722076, -267.0198364, 256.1444092
1: -12.4564848, 11.5711174, -15.7912579, 14.6004047, -27.0568886, 27.3623734
2: -7.1575665, 11.7965927, -9.0307693, 14.9529743, -22.1105404, 20.8273621
3: -5.7723017, 12.9366655, -7.2881451, 16.3678722, -22.1401749, 20.2248077
4: -8.8456087, 10.5499878, -11.1406412, 13.3778372, -22.2234440, 21.6906281

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823867
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823870
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -101.6145782, 140.8452759, -211.0228729, 195.0465088
1: -8.5443325, 8.4127283, -13.0697832, 12.1037197, -20.6480522, 21.4825096
2: -5.1585846, 8.1050158, -7.4919033, 12.3726549, -17.5312386, 15.5969191
3: -4.1931524, 8.7007360, -6.0223894, 13.5765505, -17.7696953, 14.7231245
4: -6.4483891, 7.3207774, -9.2628574, 11.0729675, -17.5213566, 16.5836353

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962044, upper bound: 20.5599406
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016765, upper bound: 20.5599435
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -101.6145782, 140.8452759, -263.1687927, 272.7122803
1: -15.8821478, 14.6396856, -13.0697832, 12.1037197, -27.9858665, 27.7094612
2: -9.0427151, 15.0180864, -7.4919033, 12.3726549, -21.4153709, 22.5099831
3: -7.2853603, 16.4737110, -6.0223894, 13.5765505, -20.8619061, 22.4961014
4: -11.1862717, 13.4413309, -9.2628574, 11.0729675, -22.2592392, 22.7041836

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962079, upper bound: 20.6055227
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016812, upper bound: 20.6055226
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -100.5943604, 139.6793518, -209.8569489, 194.0262909
1: -8.5443325, 8.4127283, -12.9418774, 12.0215712, -20.5659008, 21.3546066
2: -5.1585846, 8.1050158, -7.4400887, 12.2674332, -17.4260178, 15.5451050
3: -4.1931524, 8.7007360, -6.0007830, 13.4309692, -17.6241169, 14.7015171
4: -6.4483891, 7.3207774, -9.1817017, 10.9723797, -17.4207668, 16.5024796

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6916584, upper bound: 20.5505763
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823516, upper bound: 20.5505757
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -100.5943604, 139.6793518, -262.0028687, 271.6920471
1: -15.8821478, 14.6396856, -12.9418774, 12.0215712, -27.9037170, 27.5815620
2: -9.0427151, 15.0180864, -7.4400887, 12.2674332, -21.3101482, 22.4581757
3: -7.2853603, 16.4737110, -6.0007830, 13.4309692, -20.7163277, 22.4744949
4: -11.1862717, 13.4413309, -9.1817017, 10.9723797, -22.1586514, 22.6230278

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6916635, upper bound: 20.5969213
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823593, upper bound: 20.5969206
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -69.4079666, 92.3891907, -101.6145782, 140.8452759, -210.2532349, 194.0037689
1: -8.4298458, 8.3393698, -13.0697832, 12.1037197, -20.5335655, 21.4091492
2: -5.1099062, 8.0077209, -7.4919033, 12.3726549, -17.4825611, 15.4996214
3: -4.1703382, 8.5682316, -6.0223894, 13.5765505, -17.7468834, 14.5906210
4: -6.3853221, 7.2273917, -9.2628574, 11.0729675, -17.4582901, 16.4902496

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962003, upper bound: 20.5456326
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016750, upper bound: 20.5456360
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -121.0664978, 169.5497131, -101.6145782, 140.8452759, -261.9117737, 271.1643066
1: -15.7213001, 14.5200100, -13.0697832, 12.1037197, -27.8250179, 27.5897865
2: -8.9813738, 14.8817320, -7.4919033, 12.3726549, -21.3540268, 22.3736267
3: -7.2442417, 16.2906761, -6.0223894, 13.5765505, -20.8207855, 22.3130646
4: -11.0818872, 13.3121700, -9.2628574, 11.0729675, -22.1548519, 22.5750275

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6962061, upper bound: 20.5937058
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016792, upper bound: 20.5937066
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -69.4079666, 92.3891907, -100.5943604, 139.6793518, -209.0873108, 192.9835510
1: -8.4298458, 8.3393698, -12.9418774, 12.0215712, -20.4514160, 21.2812462
2: -5.1099062, 8.0077209, -7.4400887, 12.2674332, -17.3773384, 15.4478092
3: -4.1703382, 8.5682316, -6.0007830, 13.4309692, -17.6013050, 14.5690145
4: -6.3853221, 7.2273917, -9.1817017, 10.9723797, -17.3577023, 16.4090939

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6916546, upper bound: 20.5163725
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823502, upper bound: 20.5163698
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -121.0664978, 169.5497131, -100.5943604, 139.6793518, -260.7458496, 270.1440735
1: -15.7213001, 14.5200100, -12.9418774, 12.0215712, -27.7428703, 27.4618874
2: -8.9813738, 14.8817320, -7.4400887, 12.2674332, -21.2488041, 22.3218212
3: -7.2442417, 16.2906761, -6.0007830, 13.4309692, -20.6752071, 22.2914581
4: -11.0818872, 13.3121700, -9.1817017, 10.9723797, -22.0542622, 22.4938717

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6916642, upper bound: 20.5684636
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823523, upper bound: 20.5684564
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -123.1294250, 172.0789642, -294.4024353, 294.2271118
1: -15.8821478, 14.6396856, -15.9661789, 14.7359285, -30.6180763, 30.6058617
2: -9.0427151, 15.0180864, -9.1003685, 15.1034212, -24.1461372, 24.1184540
3: -7.2853603, 16.4737110, -7.3377576, 16.5662975, -23.8516560, 23.8114681
4: -11.1862717, 13.4413309, -11.2568188, 13.5198822, -24.7061501, 24.6981468

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5632712, upper bound: 20.5775233
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5632712, upper bound: 20.6065585
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -121.7339630, 170.3722076, -292.6956482, 292.8316040
1: -15.8821478, 14.6396856, -15.7912579, 14.6004047, -30.4825497, 30.4309425
2: -9.0427151, 15.0180864, -9.0307693, 14.9529743, -23.9956894, 24.0488548
3: -7.2853603, 16.4737110, -7.2881451, 16.3678722, -23.6532326, 23.7618561
4: -11.1862717, 13.4413309, -11.1406412, 13.3778372, -24.5641098, 24.5819683

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -20.5481382, upper bound: 20.5781156
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5481382, upper bound: 20.6065551
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962286, upper bound: 20.6962276
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962286, upper bound: 20.6962297
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962304, upper bound: 20.7017027
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962304, upper bound: 20.7017003
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6907926, upper bound: 20.6938734
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6804714, upper bound: 20.6938725
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6907933, upper bound: 20.6987052
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6804721, upper bound: 20.6987014
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6992308, upper bound: 20.6716533
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716528
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6992307, upper bound: 20.6708115
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708099
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824014
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824008
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6823972
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6824016, upper bound: 20.6824011
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6190990, upper bound: 20.6805913
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805858
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6190992, upper bound: 20.6831516
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5885817, upper bound: 20.6831457
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6055122, upper bound: 20.6938608
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5813300, upper bound: 20.6938580
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6055129, upper bound: 20.6986913
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5813307, upper bound: 20.6986907
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6116696, upper bound: 20.6716446
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716384
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6116695, upper bound: 20.6708002
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6708002
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823880
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823855
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823867
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5839509, upper bound: 20.6823870
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962044, upper bound: 20.5599406
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.7016765, upper bound: 20.5599435
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962079, upper bound: 20.6055227
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.7016812, upper bound: 20.6055226
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6916584, upper bound: 20.5505763
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6823516, upper bound: 20.5505757
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6916635, upper bound: 20.5969213
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6823593, upper bound: 20.5969206
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962003, upper bound: 20.5456326
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.7016750, upper bound: 20.5456360
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6962061, upper bound: 20.5937058
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.7016792, upper bound: 20.5937066
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6916546, upper bound: 20.5163725
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6823502, upper bound: 20.5163698
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6916642, upper bound: 20.5684636
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.6823523, upper bound: 20.5684564
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5632712, upper bound: 20.5775233
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5632712, upper bound: 20.6065585
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5481382, upper bound: 20.5781156
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 3, lower bound: -20.5481382, upper bound: 20.6065551

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -96.7078705, 135.1683655, -231.8762360, 231.8762360
1: -12.5404415, 11.6070614, -12.5404415, 11.6070614, -24.1474953, 24.1474991
2: -7.1486292, 11.8655405, -7.1486292, 11.8655405, -19.0141697, 19.0141697
3: -5.7478900, 13.0301399, -5.7478900, 13.0301399, -18.7780247, 18.7780266
4: -8.8640528, 10.6130219, -8.8640528, 10.6130219, -19.4770737, 19.4770737

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938648, upper bound: 20.6314941
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938720, upper bound: 20.6938720
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -95.0735168, 132.7543030, -229.4621735, 230.2418823
1: -12.5404415, 11.6070614, -12.3337259, 11.3693285, -23.9097652, 23.9407883
2: -7.1486292, 11.8655405, -7.0452828, 11.6368580, -18.7854881, 18.9108219
3: -5.7478900, 13.0301399, -5.6269884, 12.7925835, -18.5404682, 18.6571274
4: -8.8640528, 10.6130219, -8.6930714, 10.4093132, -19.2733631, 19.3060932

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938648, upper bound: 20.6314968
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938720, upper bound: 20.6938730
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -96.7078705, 135.1683655, -230.2418823, 229.4621735
1: -12.3337259, 11.3693285, -12.5404415, 11.6070614, -23.9407883, 23.9097672
2: -7.0452828, 11.6368580, -7.1486292, 11.8655405, -18.9108219, 18.7854881
3: -5.6269884, 12.7925835, -5.7478900, 13.0301399, -18.6571274, 18.5404663
4: -8.6930714, 10.4093132, -8.8640528, 10.6130219, -19.3060932, 19.2733631

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938726, upper bound: 20.6988927
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938729, upper bound: 20.6986991
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -95.0735168, 132.7543030, -227.8278198, 227.8278198
1: -12.3337259, 11.3693285, -12.3337259, 11.3693285, -23.7030544, 23.7030544
2: -7.0452828, 11.6368580, -7.0452828, 11.6368580, -18.6821404, 18.6821404
3: -5.6269884, 12.7925835, -5.6269884, 12.7925835, -18.4195690, 18.4195690
4: -8.6930714, 10.4093132, -8.6930714, 10.4093132, -19.1023808, 19.1023827

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938726, upper bound: 20.6988942
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6938729, upper bound: 20.6986992
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -99.1583405, 137.6558838, -234.3637543, 234.3267059
1: -12.5404415, 11.6070614, -12.7440815, 11.8651171, -24.4055576, 24.3511410
2: -7.1486292, 11.8655405, -7.3402557, 12.0819092, -19.2305374, 19.2057953
3: -5.7478900, 13.0301399, -5.9315467, 13.2485046, -18.9963894, 18.9616833
4: -8.8640528, 10.6130219, -9.0659151, 10.8102217, -19.6742725, 19.6789360

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804641, upper bound: 20.6314970
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804641, upper bound: 20.6938715
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -96.6476288, 134.4104614, -231.1183319, 231.8159943
1: -12.5404415, 11.6070614, -12.4564848, 11.5711174, -24.1115532, 24.0635433
2: -7.1486292, 11.8655405, -7.1575665, 11.7965927, -18.9452209, 19.0231056
3: -5.7478900, 13.0301399, -5.7723017, 12.9366655, -18.6845512, 18.8024406
4: -8.8640528, 10.6130219, -8.8456087, 10.5499878, -19.4140396, 19.4586296

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804641, upper bound: 20.6314979
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804641, upper bound: 20.6938711
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -99.1583405, 137.6558838, -232.7294006, 231.9126434
1: -12.3337259, 11.3693285, -12.7440815, 11.8651171, -24.1988430, 24.1134109
2: -7.0452828, 11.6368580, -7.3402557, 12.0819092, -19.1271877, 18.9771137
3: -5.6269884, 12.7925835, -5.9315467, 13.2485046, -18.8754921, 18.7241249
4: -8.6930714, 10.4093132, -9.0659151, 10.8102217, -19.5032921, 19.4752235

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804722, upper bound: 20.6987035
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804722, upper bound: 20.6986977
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -96.6476288, 134.4104614, -229.4839783, 229.4019318
1: -12.3337259, 11.3693285, -12.4564848, 11.5711174, -23.9048424, 23.8258114
2: -7.0452828, 11.6368580, -7.1575665, 11.7965927, -18.8418732, 18.7944241
3: -5.6269884, 12.7925835, -5.7723017, 12.9366655, -18.5636539, 18.5648823
4: -8.6930714, 10.4093132, -8.8456087, 10.5499878, -19.2430592, 19.2549171

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804722, upper bound: 20.6987009
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6804722, upper bound: 20.6987005
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -96.2600021, 134.4891968, -233.6475220, 233.9158783
1: -12.7440815, 11.8651171, -12.4918880, 11.5207005, -24.2647820, 24.3570061
2: -7.3402557, 12.0819092, -7.1298404, 11.7859764, -19.1262321, 19.2117500
3: -5.9315467, 13.2485046, -5.7129354, 12.9593010, -18.8908482, 18.9614410
4: -9.0659151, 10.8102217, -8.8134222, 10.5465651, -19.6124763, 19.6236439

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716499
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716515
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -98.6699753, 136.9473572, -236.1056824, 236.3258362
1: -12.7440815, 11.8651171, -12.7103720, 11.7568588, -24.5009403, 24.5754890
2: -7.3402557, 12.0819092, -7.2825561, 12.0280361, -19.3682919, 19.3644657
3: -5.9315467, 13.2485046, -5.8451309, 13.1911173, -19.1226597, 19.0936356
4: -9.0659151, 10.8102217, -9.0010395, 10.7614632, -19.8273754, 19.8112602

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716532
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822836, upper bound: 20.6716499
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -96.2600021, 134.4891968, -231.1368256, 230.6704712
1: -12.4564848, 11.5711174, -12.4918880, 11.5207005, -23.9771824, 24.0630035
2: -7.1575665, 11.7965927, -7.1298404, 11.7859764, -18.9435406, 18.9264336
3: -5.7723017, 12.9366655, -5.7129354, 12.9593010, -18.7316017, 18.6496010
4: -8.8456087, 10.5499878, -8.8134222, 10.5465651, -19.3921661, 19.3634109

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708114
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708120
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -98.6699753, 136.9473572, -233.5949860, 233.0804291
1: -12.4564848, 11.5711174, -12.7103720, 11.7568588, -24.2133427, 24.2814903
2: -7.1575665, 11.7965927, -7.2825561, 12.0280361, -19.1856022, 19.0791492
3: -5.7723017, 12.9366655, -5.8451309, 13.1911173, -18.9634190, 18.7817955
4: -8.8456087, 10.5499878, -9.0010395, 10.7614632, -19.6070709, 19.5510273

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708122
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6822835, upper bound: 20.6708110
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -99.1583405, 137.6558838, -236.8142242, 236.8142242
1: -12.7440815, 11.8651171, -12.7440815, 11.8651171, -24.6091995, 24.6091995
2: -7.3402557, 12.0819092, -7.3402557, 12.0819092, -19.4221630, 19.4221649
3: -5.9315467, 13.2485046, -5.9315467, 13.2485046, -19.1800480, 19.1800499
4: -9.0659151, 10.8102217, -9.0659151, 10.8102217, -19.8761368, 19.8761368

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708155, upper bound: 20.6917029
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708130, upper bound: 20.6716494
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -96.6476288, 134.4104614, -233.5688019, 234.3035126
1: -12.7440815, 11.8651171, -12.4564848, 11.5711174, -24.3151989, 24.3216000
2: -7.3402557, 12.0819092, -7.1575665, 11.7965927, -19.1368484, 19.2394714
3: -5.9315467, 13.2485046, -5.7723017, 12.9366655, -18.8682098, 19.0208054
4: -9.0659151, 10.8102217, -8.8456087, 10.5499878, -19.6159019, 19.6558304

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708155, upper bound: 20.6917024
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708130, upper bound: 20.6716480
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -99.1583405, 137.6558838, -234.3035126, 233.5688019
1: -12.4564848, 11.5711174, -12.7440815, 11.8651171, -24.3216000, 24.3151989
2: -7.1575665, 11.7965927, -7.3402557, 12.0819092, -19.2394733, 19.1368484
3: -5.7723017, 12.9366655, -5.9315467, 13.2485046, -19.0208054, 18.8682098
4: -8.8456087, 10.5499878, -9.0659151, 10.8102217, -19.6558304, 19.6159019

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708142, upper bound: 20.6823359
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708129, upper bound: 20.6708104
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -96.6476288, 134.4104614, -231.0580902, 231.0580902
1: -12.4564848, 11.5711174, -12.4564848, 11.5711174, -24.0275993, 24.0276012
2: -7.1575665, 11.7965927, -7.1575665, 11.7965927, -18.9541588, 18.9541588
3: -5.7723017, 12.9366655, -5.7723017, 12.9366655, -18.7089672, 18.7089672
4: -8.8456087, 10.5499878, -8.8456087, 10.5499878, -19.3955956, 19.3955956

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708142, upper bound: 20.6823357
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6708129, upper bound: 20.6708117
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -117.1653519, 164.9637146, -261.6715698, 252.3337097
1: -12.5404415, 11.6070614, -15.3191633, 14.0799551, -26.6203918, 26.9262238
2: -7.1486292, 11.8655405, -8.6932688, 14.4508619, -21.5994892, 20.5588093
3: -5.7478900, 13.0301399, -6.9871798, 15.8780527, -21.6259384, 20.0173168
4: -8.8640528, 10.6130219, -10.7562647, 12.9341917, -21.7982445, 21.3692856

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805873
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805844
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -119.6661606, 167.5651398, -264.2729492, 254.8345337
1: -12.5404415, 11.6070614, -15.5529823, 14.3277044, -26.8681431, 27.1600380
2: -7.1486292, 11.8655405, -8.8534985, 14.7057409, -21.8543682, 20.7190380
3: -5.7478900, 13.0301399, -7.1264162, 16.1234055, -21.8712959, 20.1565533
4: -8.8640528, 10.6130219, -10.9505005, 13.1603222, -22.0243740, 21.5635223

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805853
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5885816, upper bound: 20.6805846
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -117.1653519, 164.9637146, -260.0372314, 249.9196472
1: -12.3337259, 11.3693285, -15.3191633, 14.0799551, -26.4136810, 26.6884918
2: -7.0452828, 11.6368580, -8.6932688, 14.4508619, -21.4961414, 20.3301277
3: -5.6269884, 12.7925835, -6.9871798, 15.8780527, -21.5050411, 19.7797585
4: -8.6930714, 10.4093132, -10.7562647, 12.9341917, -21.6272621, 21.1655769

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6111715, upper bound: 20.6783590
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6111722, upper bound: 20.6812662
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -119.6661606, 167.5651398, -262.6385803, 252.4204559
1: -12.3337259, 11.3693285, -15.5529823, 14.3277044, -26.6614304, 26.9223099
2: -7.0452828, 11.6368580, -8.8534985, 14.7057409, -21.7510185, 20.4903564
3: -5.6269884, 12.7925835, -7.1264162, 16.1234055, -21.7503929, 19.9189968
4: -8.6930714, 10.4093132, -10.9505005, 13.1603222, -21.8533936, 21.3598099

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5862209, upper bound: 20.6783536
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5862217, upper bound: 20.6812610
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -119.7311859, 167.7318115, -264.4396973, 254.8995514
1: -12.5404415, 11.6070614, -15.5400391, 14.3805313, -26.9209728, 27.1470985
2: -7.1486292, 11.8655405, -8.8936920, 14.7116756, -21.8603020, 20.7592316
3: -5.7478900, 13.0301399, -7.1836982, 16.1275597, -21.8754482, 20.2138367
4: -8.8640528, 10.6130219, -10.9791079, 13.1634712, -22.0275230, 21.5921288

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813227, upper bound: 20.6314866
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813227, upper bound: 20.6938563
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -96.7078705, 135.1683655, -117.7652740, 165.1002808, -261.8081055, 252.9336395
1: -12.5404415, 11.6070614, -15.3072262, 14.1460114, -26.6864471, 26.9142818
2: -7.1486292, 11.8655405, -8.7479668, 14.4818630, -21.6304932, 20.6135063
3: -5.7478900, 13.0301399, -7.0564718, 15.8739948, -21.6218815, 20.0866070
4: -8.8640528, 10.6130219, -10.8037090, 12.9554806, -21.8195305, 21.4167309

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813227, upper bound: 20.6314862
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813227, upper bound: 20.6938552
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -119.7311859, 167.7318115, -262.8053284, 252.4854889
1: -12.3337259, 11.3693285, -15.5400391, 14.3805313, -26.7142563, 26.9093666
2: -7.0452828, 11.6368580, -8.8936920, 14.7116756, -21.7569542, 20.5305500
3: -5.6269884, 12.7925835, -7.1836982, 16.1275597, -21.7545471, 19.9762802
4: -8.6930714, 10.4093132, -10.9791079, 13.1634712, -21.8565426, 21.3884182

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813308, upper bound: 20.6986886
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813308, upper bound: 20.6986895
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -95.0735168, 132.7543030, -117.7652740, 165.1002808, -260.1737671, 250.5195465
1: -12.3337259, 11.3693285, -15.3072262, 14.1460114, -26.4797344, 26.6765518
2: -7.0452828, 11.6368580, -8.7479668, 14.4818630, -21.5271435, 20.3848248
3: -5.6269884, 12.7925835, -7.0564718, 15.8739948, -21.5009823, 19.8490505
4: -8.6930714, 10.4093132, -10.8037090, 12.9554806, -21.6485500, 21.2130222

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813308, upper bound: 20.6986888
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5813304, upper bound: 20.6986866
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -117.1653519, 164.9637146, -264.1220703, 254.8212280
1: -12.7440815, 11.8651171, -15.3191633, 14.0799551, -26.8240356, 27.1842804
2: -7.3402557, 12.0819092, -8.6932688, 14.4508619, -21.7911186, 20.7751770
3: -5.9315467, 13.2485046, -6.9871798, 15.8780527, -21.8095970, 20.2356815
4: -9.0659151, 10.8102217, -10.7562647, 12.9341917, -22.0001068, 21.5664864

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716405
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716388
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -119.6661606, 167.5651398, -266.7234802, 257.3220520
1: -12.7440815, 11.8651171, -15.5529823, 14.3277044, -27.0717850, 27.4180984
2: -7.3402557, 12.0819092, -8.8534985, 14.7057409, -22.0459938, 20.9354076
3: -5.9315467, 13.2485046, -7.1264162, 16.1234055, -22.0549526, 20.3749180
4: -9.0659151, 10.8102217, -10.9505005, 13.1603222, -22.2262363, 21.7607212

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716399
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886408, upper bound: 20.6716362
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -117.1653519, 164.9637146, -261.6113281, 251.5758057
1: -12.4564848, 11.5711174, -15.3191633, 14.0799551, -26.5364380, 26.8902817
2: -7.1575665, 11.7965927, -8.6932688, 14.4508619, -21.6084270, 20.4898605
3: -5.7723017, 12.9366655, -6.9871798, 15.8780527, -21.6503544, 19.9238453
4: -8.8456087, 10.5499878, -10.7562647, 12.9341917, -21.7798004, 21.3062515

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6707998
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6707999
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -119.6661606, 167.5651398, -264.2127686, 254.0766296
1: -12.4564848, 11.5711174, -15.5529823, 14.3277044, -26.7841854, 27.1240959
2: -7.1575665, 11.7965927, -8.8534985, 14.7057409, -21.8633060, 20.6500912
3: -5.7723017, 12.9366655, -7.1264162, 16.1234055, -21.8957062, 20.0630817
4: -8.8456087, 10.5499878, -10.9505005, 13.1603222, -22.0059299, 21.5004883

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6708003
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5886407, upper bound: 20.6707999
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -119.7311859, 167.7318115, -266.8901367, 257.3870850
1: -12.7440815, 11.8651171, -15.5400391, 14.3805313, -27.1246128, 27.4051552
2: -7.3402557, 12.0819092, -8.8936920, 14.7116756, -22.0519295, 20.9756012
3: -5.9315467, 13.2485046, -7.1836982, 16.1275597, -22.0591030, 20.4322014
4: -9.0659151, 10.8102217, -10.9791079, 13.1634712, -22.2293854, 21.7893295

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743517, upper bound: 20.6916873
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743492, upper bound: 20.6716377
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -99.1583405, 137.6558838, -117.7652740, 165.1002808, -264.2586060, 255.4211578
1: -12.7440815, 11.8651171, -15.3072262, 14.1460114, -26.8900909, 27.1723423
2: -7.3402557, 12.0819092, -8.7479668, 14.4818630, -21.8221188, 20.8298759
3: -5.9315467, 13.2485046, -7.0564718, 15.8739948, -21.8055382, 20.3049736
4: -9.0659151, 10.8102217, -10.8037090, 12.9554806, -22.0213909, 21.6139297

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743517, upper bound: 20.6916902
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743492, upper bound: 20.6716386
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -119.7311859, 167.7318115, -264.3794556, 254.1416473
1: -12.4564848, 11.5711174, -15.5400391, 14.3805313, -26.8370152, 27.1111565
2: -7.1575665, 11.7965927, -8.8936920, 14.7116756, -21.8692398, 20.6902847
3: -5.7723017, 12.9366655, -7.1836982, 16.1275597, -21.8998604, 20.1203632
4: -8.8456087, 10.5499878, -10.9791079, 13.1634712, -22.0090790, 21.5290947

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743504, upper bound: 20.6823260
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743491, upper bound: 20.6707944
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -96.6476288, 134.4104614, -117.7652740, 165.1002808, -261.7479248, 252.1757050
1: -12.4564848, 11.5711174, -15.3072262, 14.1460114, -26.6024895, 26.8783398
2: -7.1575665, 11.7965927, -8.7479668, 14.4818630, -21.6394291, 20.5445595
3: -5.7723017, 12.9366655, -7.0564718, 15.8739948, -21.6462955, 19.9931355
4: -8.8456087, 10.5499878, -10.8037090, 12.9554806, -21.8010845, 21.3536968

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743504, upper bound: 20.6823264
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.5743491, upper bound: 20.6707976
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -96.7078705, 135.1683655, -205.3459625, 190.1398010
1: -8.5443325, 8.4127283, -12.5404415, 11.6070614, -20.1513939, 20.9531670
2: -5.1585846, 8.1050158, -7.1486292, 11.8655405, -17.0241241, 15.2536449
3: -4.1931524, 8.7007360, -5.7478900, 13.0301399, -17.2232876, 14.4486256
4: -6.4483891, 7.3207774, -8.8640528, 10.6130219, -17.0614109, 16.1848297

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6961996, upper bound: 20.5528276
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6961996, upper bound: 20.5599391
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -95.0735168, 132.7543030, -202.9319000, 188.5054474
1: -8.5443325, 8.4127283, -12.3337259, 11.3693285, -19.9136620, 20.7464542
2: -5.1585846, 8.1050158, -7.0452828, 11.6368580, -16.7954426, 15.1502991
3: -4.1931524, 8.7007360, -5.6269884, 12.7925835, -16.9857273, 14.3277245
4: -6.4483891, 7.3207774, -8.6930714, 10.4093132, -16.8577003, 16.0138493

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016752, upper bound: 20.5528293
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.7016752, upper bound: 20.5599430
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -96.7078705, 135.1683655, -257.4918823, 267.8055115
1: -15.8821478, 14.6396856, -12.5404415, 11.6070614, -27.4892082, 27.1801224
2: -9.0427151, 15.0180864, -7.1486292, 11.8655405, -20.9082565, 22.1667118
3: -7.2853603, 16.4737110, -5.7478900, 13.0301399, -20.3154964, 22.2216015
4: -11.1862717, 13.4413309, -8.8640528, 10.6130219, -21.7992935, 22.3053799

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6805799, upper bound: 20.6054488
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6805743, upper bound: 20.5718326
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -95.0735168, 132.7543030, -255.0778046, 266.1711426
1: -15.8821478, 14.6396856, -12.3337259, 11.3693285, -27.2514763, 26.9734116
2: -9.0427151, 15.0180864, -7.0452828, 11.6368580, -20.6795731, 22.0633659
3: -7.2853603, 16.4737110, -5.6269884, 12.7925835, -20.0779400, 22.1007004
4: -11.1862717, 13.4413309, -8.6930714, 10.4093132, -21.5955849, 22.1343975

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6805805, upper bound: 20.6054481
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6831288, upper bound: 20.5718365
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -99.1583405, 137.6558838, -207.8334808, 192.5902710
1: -8.5443325, 8.4127283, -12.7440815, 11.8651171, -20.4094505, 21.1568108
2: -5.1585846, 8.1050158, -7.3402557, 12.0819092, -17.2404938, 15.4452696
3: -4.1931524, 8.7007360, -5.9315467, 13.2485046, -17.4416523, 14.6322823
4: -6.4483891, 7.3207774, -9.0659151, 10.8102217, -17.2586098, 16.3866920

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823546, upper bound: 20.5505791
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823546, upper bound: 20.5505787
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -70.1775970, 93.4319305, -96.6476288, 134.4104614, -204.5880585, 190.0795593
1: -8.5443325, 8.4127283, -12.4564848, 11.5711174, -20.1154480, 20.8692112
2: -5.1585846, 8.1050158, -7.1575665, 11.7965927, -16.9551773, 15.2625799
3: -4.1931524, 8.7007360, -5.7723017, 12.9366655, -17.1298141, 14.4730368
4: -6.4483891, 7.3207774, -8.8456087, 10.5499878, -16.9983768, 16.1663857

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823546, upper bound: 20.5505773
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6823546, upper bound: 20.5505765
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -99.1583405, 137.6558838, -259.9793701, 270.2560425
1: -15.8821478, 14.6396856, -12.7440815, 11.8651171, -27.7472649, 27.3837662
2: -9.0427151, 15.0180864, -7.3402557, 12.0819092, -21.1246243, 22.3583412
3: -7.2853603, 16.4737110, -5.9315467, 13.2485046, -20.5338612, 22.4052582
4: -11.1862717, 13.4413309, -9.0659151, 10.8102217, -21.9964943, 22.5072403

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6716198, upper bound: 20.5967123
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6716110, upper bound: 20.5714127
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -122.3235168, 171.0977478, -96.6476288, 134.4104614, -256.7339783, 267.7453308
1: -15.8821478, 14.6396856, -12.4564848, 11.5711174, -27.4532661, 27.0961666
2: -9.0427151, 15.0180864, -7.1575665, 11.7965927, -20.8393078, 22.1756516
3: -7.2853603, 16.4737110, -5.7723017, 12.9366655, -20.2220249, 22.2460136
4: -11.1862717, 13.4413309, -8.8456087, 10.5499878, -21.7362595, 22.2869339

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6707742, upper bound: 20.5967109
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -20.6707698, upper bound: 20.5714077
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -69.4079666, 92.3891907, -96.7078705, 135.1683655, -204.5763245, 189.0970612
1: -8.4298458, 8.3393698, -12.5404415, 11.6070614, -20.0369072, 20.8798103
2: -5.1099062, 8.0077209, -7.1486292, 11.8655405, -16.9754467, 15.1563501
3: -4.1703382, 8.5682316, -5.7478900, 13.0301399, -17.2004738, 14.3161221
4: -6.3853221, 7.2273917, -8.8640528, 10.6130219, -16.9983444, 16.0914440

Time for backsubstitution: 2.78 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.35 + 417.94 = 422.29 seconds
