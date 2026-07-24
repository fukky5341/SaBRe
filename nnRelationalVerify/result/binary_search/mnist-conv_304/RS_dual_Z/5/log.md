## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.3532293525
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.8783393, 3.8783391)
1: (-12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.5584154, 3.5584154)
2: (-13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.2301512, 3.2301512)
3: (-9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -2.1610408, 2.1610408)
5: (-11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.7072897, 3.7072897)
6: (-17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.9770737, 3.9770737)
7: (-6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.8377733, 2.8377733)
8: (-2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.2236829, 2.2236829)
9: (2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.7430749, 2.7430749)

## BASE Result
execution time: IAR + LP analysis = 15.56 + 35.74 = 51.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1375822, upper bound: 2.1375790


# Binary Search by BASE starts (time budget: 3548.70 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.467238664627075
rel_dist={9: [-1.6640502761084588, 1.6640497405138106]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.188438653945923
rel_dist={9: [-1.0985895441039362, 1.0985873760165363]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.244198799133301
rel_dist={9: [-1.2428127078072388, 1.2428113778761434]}

## Binary Search Result
Binary search time: 218.17 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3330.53 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7511097, upper bound: 1.7535539
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535517, upper bound: 1.7511119
time: 4.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.97
Output dim: 9, lower bound: -1.7511097, upper bound: 1.7535539
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.97
Output dim: 9, lower bound: -1.7535517, upper bound: 1.7511119

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1256166, 3.1300211
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0381641, 3.0402875
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0089107, 3.0217595
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9808764
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8732660, 1.8772860
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0818176, 3.0656190
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3747044, 3.3766141
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5084982, 2.5131974
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0381880, 2.0316546
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5197005, 2.5223250

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7496628, upper bound: 1.7535468
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7521070
time: 4.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1300206, 3.1256156
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0402880, 3.0381637
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0217595, 3.0089111
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9808755, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8772857, 1.8732662
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0656185, 3.0818186
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3766136, 3.3747046
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5131974, 2.5084982
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0316544, 2.0381882
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5223250, 2.5197005

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7521048, upper bound: 1.7511050
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7496650
time: 4.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.64
Output dim: 9, lower bound: -1.7496628, upper bound: 1.7535468
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.64
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7521070
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.64
Output dim: 9, lower bound: -1.7521048, upper bound: 1.7511050
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.64
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7496650

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1186132, 3.1097870
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0346937, 3.0303392
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0020928, 3.0193925
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9806986
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8687189, 1.8641303
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0739431, 3.0628786
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3671598, 3.3548307
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5083823, 2.5128787
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0366716, 2.0272791
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5147457, 2.5206113

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7496273, upper bound: 1.7335386
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296547, upper bound: 1.7535114
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1053810, 3.1230187
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0282154, 3.0368171
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0065436, 3.0149412
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9803724
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8601110, 1.8727376
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0790749, 3.0577435
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3529215, 3.3690686
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5081797, 2.5130813
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0338125, 2.0301371
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5179868, 2.5173700

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7510696, upper bound: 1.7320963
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310972, upper bound: 1.7520694
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1230192, 3.1053810
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0368166, 3.0282154
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0149417, 3.0065446
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9803720, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8727376, 1.8601105
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0577431, 3.0790753
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3690691, 3.3529212
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5130811, 2.5081797
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0301371, 2.0338128
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5173697, 2.5179870

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7520693, upper bound: 1.7310971
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7320967, upper bound: 1.7510694
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1097870, 3.1186132
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0303392, 3.0346932
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0193925, 3.0020928
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9806981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8641307, 1.8687189
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0628786, 3.0739431
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3548307, 3.3671591
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5128789, 2.5083823
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0272789, 2.0366719
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5206113, 2.5147457

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535115, upper bound: 1.7296569
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7335390, upper bound: 1.7496274
time: 4.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7496273, upper bound: 1.7335386
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7296547, upper bound: 1.7535114
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7510696, upper bound: 1.7320963
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7310972, upper bound: 1.7520694
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7520693, upper bound: 1.7310971
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7320967, upper bound: 1.7510694
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7535115, upper bound: 1.7296569
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.80
Output dim: 9, lower bound: -1.7335390, upper bound: 1.7496274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1208763, 3.1147041
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0024796, 3.0075235
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0144830, 3.0393677
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8155198, 1.8264787
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0831919, 3.0748668
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3493853, 3.3422408
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4994664, 2.4947910
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0240974, 2.0095291
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4968123, 2.4952965

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493543, upper bound: 1.7311455
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272729, upper bound: 1.7311600
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1235304, 3.1120491
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0118780, 2.9981251
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0220675, 3.0317826
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8310676, 1.8109310
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0859308, 3.0721283
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3545685, 3.3370566
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4902945, 2.5039630
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0189219, 2.0147047
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4894309, 2.5026777

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272758, upper bound: 1.7311572
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272590, upper bound: 1.7532252
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1076431, 3.1279364
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9960022, 3.0140014
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0189338, 3.0349164
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8069115, 1.8350861
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0883245, 3.0697317
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3351469, 3.3564787
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4992642, 2.4949937
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0212383, 2.0123868
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5000534, 2.4920552

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7507812, upper bound: 1.7297011
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7287154, upper bound: 1.7297177
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1102982, 3.1252813
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0053997, 3.0046029
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0265193, 3.0273314
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8224592, 1.8195384
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0910635, 3.0669932
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3403311, 3.3512945
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4900918, 2.5041656
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0160627, 2.0175624
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4926724, 2.4994361

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7287182, upper bound: 1.7297151
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7287015, upper bound: 1.7517986
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1252813, 3.1102986
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0046034, 3.0054002
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0273318, 3.0265193
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8195381, 1.8224590
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0669928, 3.0910635
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3512945, 3.3403313
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5041656, 2.4900918
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0175624, 2.0160627
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4994364, 2.4926722

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517984, upper bound: 1.7287037
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297149, upper bound: 1.7287182
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1279354, 3.1076436
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0140018, 2.9960012
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0349154, 3.0189347
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8350863, 1.8069112
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0697317, 3.0883250
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3564787, 3.3351471
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4949937, 2.4992640
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0123873, 2.0212381
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4920554, 2.5000532

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297178, upper bound: 1.7287154
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297010, upper bound: 1.7507812
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1120491, 3.1235304
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9981251, 3.0118780
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0317826, 3.0220680
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8109312, 1.8310673
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0721292, 3.0859313
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3370571, 3.3545690
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5039630, 2.4902945
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0147047, 2.0189219
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5026779, 2.4894309

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532252, upper bound: 1.7272591
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7311572, upper bound: 1.7272756
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1147041, 3.1208758
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0075245, 3.0024796
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0393672, 3.0144830
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8264790, 1.8155196
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0748663, 3.0831928
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3422413, 3.3493850
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4947910, 2.4994664
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0095291, 2.0240972
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4952970, 2.4968119

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7311601, upper bound: 1.7272731
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7311433, upper bound: 1.7493546
time: 4.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7493543, upper bound: 1.7311455
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7272729, upper bound: 1.7311600
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7272758, upper bound: 1.7311572
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7272590, upper bound: 1.7532252
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7507812, upper bound: 1.7297011
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7287154, upper bound: 1.7297177
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7287182, upper bound: 1.7297151
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7287015, upper bound: 1.7517986
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7517984, upper bound: 1.7287037
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7297149, upper bound: 1.7287182
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7297178, upper bound: 1.7287154
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7297010, upper bound: 1.7507812
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7532252, upper bound: 1.7272591
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7311572, upper bound: 1.7272756
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7311601, upper bound: 1.7272731
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.34
Output dim: 9, lower bound: -1.7311433, upper bound: 1.7493546

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1230316, 3.1176729
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9956913, 3.0027161
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0193148, 3.0460229
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8034811, 1.8185527
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0847340, 3.0769892
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3446064, 3.3388548
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5071526, 2.4997153
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0214176, 2.0057459
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4919224, 2.4883924

Time for backsubstitution: 13.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493446, upper bound: 1.7310677
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7491902, upper bound: 1.7296947
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1238451, 3.1168599
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9976730, 3.0007358
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0211334, 3.0441990
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8075943, 1.8144403
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0853148, 3.0764079
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3459988, 3.3374629
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5043907, 2.5024648
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0203142, 2.0068488
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4899077, 2.4904068

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272632, upper bound: 1.7310843
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1256866, 3.1150184
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0050888, 2.9933181
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0268993, 3.0384336
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8190289, 1.8030062
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0874720, 3.0742507
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3497906, 3.3336706
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4979687, 2.5088873
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0162420, 2.0109212
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4845409, 2.4957736

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272661, upper bound: 1.7310816
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1265001, 3.1142054
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0070705, 2.9913373
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0287228, 3.0366139
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8231411, 1.7988925
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0880527, 3.0736694
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3511829, 3.3322787
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4952192, 2.5116494
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0151386, 2.0120244
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4825268, 2.4977877

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1097994, 3.1309052
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9892139, 3.0091939
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0237656, 3.0415716
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.7948728, 1.8271601
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0898666, 3.0718536
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3303699, 3.3530927
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5069504, 2.4999180
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0185575, 2.0086038
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4951634, 2.4851511

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1106129, 3.1300921
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9911938, 3.0072131
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0255852, 3.0397477
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.7989864, 1.8230476
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0904465, 3.0712729
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3317604, 3.3517005
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5041885, 2.5026674
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0174551, 2.0097067
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4931488, 2.4871655

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1124544, 3.1282501
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9986115, 2.9997959
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0313511, 3.0339823
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8104205, 1.8116136
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0926046, 3.0691156
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3355541, 3.3479085
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4977655, 2.5090899
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0133829, 2.0137792
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4877825, 2.4925320

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1132669, 3.1274371
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0005913, 2.9978147
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0331745, 3.0321627
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8145328, 1.8074999
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0931854, 3.0685344
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3369446, 3.3465166
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4950161, 2.5118520
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0122795, 2.0148821
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4857683, 2.4945464

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1274366, 3.1132674
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9978142, 3.0005922
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0321627, 3.0331750
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8074999, 1.8145330
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0685349, 3.0931854
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3465176, 3.3369451
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5118523, 2.4950163
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0148821, 2.0122795
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4945464, 2.4857681

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517887, upper bound: 1.7286260
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1282501, 3.1124544
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9997959, 2.9986119
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0339823, 3.0313511
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8116131, 1.8104205
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0691147, 3.0926046
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3479080, 3.3355532
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5090904, 2.4977655
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0137796, 2.0133824
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4925318, 2.4877825

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1300917, 3.1106129
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0072136, 2.9911942
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0397472, 3.0255857
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8230476, 1.7989864
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0712729, 3.0904474
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3516998, 3.3317609
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5026674, 2.5041883
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0097065, 2.0174549
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4871655, 2.4931490

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1309052, 3.1097994
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0091934, 2.9892135
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0415716, 3.0237660
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8271599, 1.7948728
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0718536, 3.0898662
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3530922, 3.3303690
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4999180, 2.5069501
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0086040, 2.0185578
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4851513, 2.4951634

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1142054, 3.1264997
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9913368, 3.0070701
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0366144, 3.0287232
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.7988925, 1.8231413
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0736694, 3.0880537
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3322792, 3.3511829
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5116491, 2.4952190
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0120249, 2.0151386
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4977875, 2.4825268

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1150179, 3.1256866
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9933186, 3.0050893
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0384331, 3.0268998
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8030062, 1.8190289
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0742502, 3.0874724
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3336716, 3.3497910
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5088873, 2.4979682
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0109215, 2.0162416
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4957733, 2.4845409

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1168604, 3.1238446
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0007362, 2.9976721
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0441990, 3.0211339
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8144403, 1.8075948
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0764074, 3.0853152
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3374634, 3.3459988
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5024652, 2.5043910
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0068493, 2.0203140
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4904070, 2.4899077

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1176729, 3.1230316
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0027161, 2.9956908
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0460224, 3.0193148
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8185525, 1.8034811
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0769882, 3.0847340
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3388557, 3.3446069
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4997158, 2.5071528
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0057459, 2.0214171
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4883924, 2.4919221

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441
time: 5.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7493446, upper bound: 1.7310677
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7491902, upper bound: 1.7296947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272632, upper bound: 1.7310843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272661, upper bound: 1.7310816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7517887, upper bound: 1.7286260
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.68
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1234970, 3.1146293
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9958544, 3.0016222
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0164690, 3.0464706
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8037133, 1.8170676
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0839224, 3.0771132
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3451762, 3.3351192
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5072217, 2.4992638
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0215192, 2.0050724
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4906011, 2.4885941

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.43 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6946201, upper bound: 1.6766452
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6950601, upper bound: 1.6762045
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1199875, 3.1176729
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9945974, 3.0027161
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0193148, 3.0431771
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8019958, 1.8185527
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0847340, 3.0761776
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3408713, 3.3388548
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5067010, 2.4997153
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0207429, 2.0057459
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4919224, 2.4870713

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.43 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6923743, upper bound: 1.6751051
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6923743, upper bound: 1.6746627
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1243095, 3.1138163
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9978361, 2.9996419
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0182877, 3.0446467
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8078275, 1.8129551
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0845032, 3.0765319
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3465686, 3.3337274
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5044599, 2.5020132
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0204158, 2.0061753
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4885864, 2.4906085

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.44 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6724927, upper bound: 1.6766614
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6729329, upper bound: 1.6762207
time: 4.91 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6946201, upper bound: 1.6766452
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6950601, upper bound: 1.6762045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6923743, upper bound: 1.6751051
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6923743, upper bound: 1.6746627
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6724927, upper bound: 1.6766614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.11
Output dim: 9, lower bound: -1.6729329, upper bound: 1.6762207
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7272661, upper bound: 1.7310816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7517887, upper bound: 1.7286260
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.11
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.522998809814453
rel_dist={9: [-1.7535540543698414, 1.753553758870325]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4665492, upper bound: 1.4681588
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681588, upper bound: 1.4665493
time: 5.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.87
Output dim: 9, lower bound: -1.4665492, upper bound: 1.4681588
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.87
Output dim: 9, lower bound: -1.4681588, upper bound: 1.4665493

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6773000, 2.6798167
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6798162, 2.6810293
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6505027, 2.6578441
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7774453, 2.7722373
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6586568, 1.6609538
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6813173, 2.6720610
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9907227, 2.9918137
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2664719, 2.2691572
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8355799, 1.8318465
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3524203, 2.3539202

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655862, upper bound: 1.4681542
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4665449, upper bound: 1.4671960
time: 5.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6798177, 2.6772990
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6810293, 2.6798158
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6578441, 2.6505022
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7722363, 2.7774458
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6609538, 1.6586568
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6720600, 2.6813178
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9918137, 2.9907222
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2691569, 2.2664719
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8318462, 1.8355799
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3539200, 2.3524208

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4671963, upper bound: 1.4665448
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681544, upper bound: 1.4655886
time: 5.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 9, lower bound: -1.4655862, upper bound: 1.4681542
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 9, lower bound: -1.4665449, upper bound: 1.4671960
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 9, lower bound: -1.4671963, upper bound: 1.4665448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 9, lower bound: -1.4681544, upper bound: 1.4655886

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6646261, 2.6595821
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6735697, 2.6710811
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6436839, 2.6535697
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7769418, 2.7719197
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6504200, 1.6477983
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6734428, 2.6671200
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9770746, 2.9700303
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2662692, 2.2688384
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8328381, 1.8274710
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3474655, 2.3508174

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655662, upper bound: 1.4566972
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4541288, upper bound: 1.4681342
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6570654, 2.6671433
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6698675, 2.6747823
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6462283, 2.6510258
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7771287, 2.7717333
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6455009, 1.6527169
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6763744, 2.6641855
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9689398, 2.9781661
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2661529, 2.2689543
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8312044, 1.8291042
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3493180, 2.3489652

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4665248, upper bound: 1.4557386
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4550874, upper bound: 1.4671759
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6671438, 2.6570649
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6747828, 2.6698675
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6510253, 2.6462274
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7717328, 2.7771282
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6527164, 1.6455014
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6641855, 2.6763754
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9781656, 2.9689388
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2689543, 2.2661531
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8291044, 1.8312044
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3489652, 2.3493178

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4671762, upper bound: 1.4550874
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4557388, upper bound: 1.4665245
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6595821, 2.6646261
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6710806, 2.6735692
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6535697, 2.6436839
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7719197, 2.7769418
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6477983, 1.6504204
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6671200, 2.6734428
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9700308, 2.9770746
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2688384, 2.2662690
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8274708, 1.8328383
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3508172, 2.3474655

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681343, upper bound: 1.4541286
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4566974, upper bound: 1.4655661
time: 5.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4655662, upper bound: 1.4566972
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4541288, upper bound: 1.4681342
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4665248, upper bound: 1.4557386
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4550874, upper bound: 1.4671759
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4671762, upper bound: 1.4550874
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4557388, upper bound: 1.4665245
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4681343, upper bound: 1.4541286
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 9, lower bound: -1.4566974, upper bound: 1.4655661

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6668882, 2.6633615
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6413555, 2.6442375
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6560740, 2.6702938
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7974744, 2.7894816
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5972214, 1.6034834
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6826925, 2.6779346
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9593010, 2.9552183
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2534223, 2.2507508
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8180456, 1.8097210
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3263688, 2.3255026

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4652804, upper bound: 1.4551971
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526382, upper bound: 1.4552087
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6684055, 2.6618447
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6467257, 2.6388674
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6604075, 2.6659598
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7945037, 2.7924523
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6061058, 1.5945990
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6842566, 2.6763697
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9622631, 2.9522562
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2481813, 2.2559919
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8150883, 1.8126783
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3221512, 2.3297205

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526403, upper bound: 1.4552052
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526303, upper bound: 1.4678458
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6593275, 2.6709232
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6376534, 2.6479392
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6586185, 2.6677499
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7976613, 2.7892957
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5923023, 1.6084020
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6856241, 2.6750002
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9511642, 2.9633541
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2533069, 2.2508667
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8164120, 1.8113539
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3282208, 2.3236504

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662391, upper bound: 1.4542405
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535969, upper bound: 1.4542498
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6608448, 2.6694059
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6430244, 2.6425686
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6629519, 2.6634159
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7946897, 2.7922664
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6011868, 1.5995176
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6871901, 2.6734352
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9541273, 2.9603920
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2480655, 2.2561078
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8134546, 1.8143113
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3240032, 2.3278682

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535989, upper bound: 1.4542480
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535890, upper bound: 1.4668903
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6694059, 2.6608443
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6425686, 2.6430244
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6634154, 2.6629519
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7922664, 2.7946901
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5995178, 1.6011865
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6734343, 2.6871896
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9603920, 2.9541273
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2561078, 2.2480655
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8143115, 1.8134544
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3278685, 2.3240030

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668904, upper bound: 1.4535888
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542483, upper bound: 1.4536002
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6709232, 2.6593275
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6479397, 2.6376534
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6677508, 2.6586175
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7892957, 2.7976608
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6084023, 1.5923021
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6750002, 2.6856251
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9633541, 2.9511647
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2508664, 2.2533066
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8113542, 1.8164117
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3236508, 2.3282208

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542503, upper bound: 1.4535968
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542404, upper bound: 1.4662390
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6618443, 2.6684055
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6388674, 2.6467257
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6659598, 2.6604085
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7924523, 2.7945037
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5945988, 1.6061056
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6763697, 2.6842570
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9522562, 2.9622631
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2559919, 2.2481813
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8126783, 1.8150883
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3297210, 2.3221507

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678459, upper bound: 1.4526302
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552052, upper bound: 1.4526415
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6633615, 2.6668882
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6442375, 2.6413550
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6702933, 2.6560740
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7894816, 2.7974744
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6034832, 1.5972211
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6779337, 2.6826921
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9552183, 2.9593005
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2507510, 2.2534225
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8097210, 1.8180456
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3255029, 2.3263686

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552072, upper bound: 1.4526378
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4551973, upper bound: 1.4652804
time: 4.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4652804, upper bound: 1.4551971
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4526382, upper bound: 1.4552087
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4526403, upper bound: 1.4552052
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4526303, upper bound: 1.4678458
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4662391, upper bound: 1.4542405
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4535969, upper bound: 1.4542498
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4535989, upper bound: 1.4542480
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4535890, upper bound: 1.4668903
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4668904, upper bound: 1.4535888
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4542483, upper bound: 1.4536002
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4542503, upper bound: 1.4535968
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4542404, upper bound: 1.4662390
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4678459, upper bound: 1.4526302
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4552052, upper bound: 1.4526415
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4552072, upper bound: 1.4526378
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.4551973, upper bound: 1.4652804

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6690445, 2.6659822
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6345673, 2.6385813
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6609058, 2.6761675
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8030286, 2.7940507
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5851822, 1.5937948
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6842327, 2.6798077
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9545231, 2.9512353
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2599249, 2.2556751
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8148928, 1.8059375
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3206158, 2.3185985

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4652710, upper bound: 1.4551920
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4649901, upper bound: 1.4542335
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6695089, 2.6655178
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6356983, 2.6374497
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6619453, 2.6751256
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8020434, 2.7950315
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5875330, 1.5914450
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6845646, 2.6794758
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9553185, 2.9504399
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2583466, 2.2572463
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8142624, 1.8065679
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3194647, 2.3197496

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526288, upper bound: 1.4552018
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523472, upper bound: 1.4542461
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6705618, 2.6644654
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6399384, 2.6332107
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6652393, 2.6718307
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8000531, 2.7970219
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5940666, 1.5849111
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6857977, 2.6782427
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9574852, 2.9482732
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2546768, 2.2609162
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8119354, 1.8088951
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3163977, 2.3228164

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526309, upper bound: 1.4551998
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523493, upper bound: 1.4542414
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6710262, 2.6640005
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6410694, 2.6320786
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6662817, 2.6707911
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7990727, 2.7980061
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5964170, 1.5825605
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6861305, 2.6779108
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9582806, 2.9474778
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2531061, 2.2624946
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8113050, 1.8095253
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3152466, 2.3239675

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526209, upper bound: 1.4678405
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523393, upper bound: 1.4668836
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6614828, 2.6735435
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6308651, 2.6422830
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6634493, 2.6736240
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8032146, 2.7938643
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5802631, 1.5987134
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6871662, 2.6768732
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9463863, 2.9593711
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2598095, 2.2557909
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8132591, 1.8075707
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3224678, 2.3167462

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4652738, upper bound: 1.4540519
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662338, upper bound: 1.4542308
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6619482, 2.6730790
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6319981, 2.6411510
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6644888, 2.6725817
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8022294, 2.7948451
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5826139, 1.5963635
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6874981, 2.6765413
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9471817, 2.9585757
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2582312, 2.2573619
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8126287, 1.8082011
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3213167, 2.3178973

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526316, upper bound: 1.4540634
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535916, upper bound: 1.4542409
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6630001, 2.6720266
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6362362, 2.6369123
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6677837, 2.6692872
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8002400, 2.7968354
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5891476, 1.5898294
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6887302, 2.6753082
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9493484, 2.9564090
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2545614, 2.2610319
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8103018, 1.8105280
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3182497, 2.3209641

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526337, upper bound: 1.4540600
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535937, upper bound: 1.4542388
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6634645, 2.6715617
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6373672, 2.6357803
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6688251, 2.6682477
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7992597, 2.7978201
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5914979, 1.5874789
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6890621, 2.6749763
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9501438, 2.9556136
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2529898, 2.2626102
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8096714, 1.8111584
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3170986, 2.3221152

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526237, upper bound: 1.4667047
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535837, upper bound: 1.4668810
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6715612, 2.6634650
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6357803, 2.6373677
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6682472, 2.6688256
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7978196, 2.7992592
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5874786, 1.5914979
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6749763, 2.6890631
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9556141, 2.9501443
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2626104, 2.2529898
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8111582, 1.8096712
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3221149, 2.3170989

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668812, upper bound: 1.4535837
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4667023, upper bound: 1.4526233
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6720266, 2.6630001
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6369133, 2.6362357
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6692867, 2.6677837
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7968354, 2.8002396
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5898294, 1.5891480
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6753082, 2.6887307
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9564095, 2.9493489
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2610321, 2.2545609
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8105278, 1.8103015
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3209639, 2.3182499

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542390, upper bound: 1.4535936
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540601, upper bound: 1.4526350
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6730795, 2.6619477
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6411514, 2.6319976
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6725817, 2.6644893
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7948451, 2.8022299
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5963631, 1.5826139
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6765404, 2.6874981
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9585762, 2.9471822
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2573624, 2.2582309
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8082008, 1.8126285
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3178973, 2.3213167

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542411, upper bound: 1.4535915
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540622, upper bound: 1.4526316
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6735430, 2.6614833
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6422825, 2.6308656
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6736240, 2.6634493
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7938647, 2.8032146
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5987134, 1.5802636
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6768732, 2.6871662
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9593716, 2.9463868
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2557907, 2.2598093
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8075705, 1.8132589
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3167462, 2.3224678

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542311, upper bound: 1.4662337
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540522, upper bound: 1.4652735
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6640005, 2.6710262
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6320782, 2.6410689
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6707907, 2.6662822
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7980065, 2.7990727
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5825605, 1.5964170
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6779108, 2.6861300
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9474773, 2.9582801
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2624950, 2.2531056
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8095255, 1.8113048
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3239675, 2.3152466

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668839, upper bound: 1.4523390
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678407, upper bound: 1.4526209
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6644659, 2.6705613
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6332111, 2.6399374
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6718311, 2.6652398
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7970214, 2.8000536
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5849109, 1.5940671
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6782427, 2.6857982
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9482727, 2.9574847
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2609167, 2.2546766
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8088951, 1.8119352
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3228164, 2.3163977

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542417, upper bound: 1.4523505
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552000, upper bound: 1.4526322
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6655178, 2.6695089
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6374493, 2.6356988
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6751251, 2.6619453
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7950311, 2.8020439
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5914450, 1.5875330
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6794758, 2.6845655
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9504395, 2.9553180
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2572460, 2.2583468
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8065681, 1.8142624
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3197498, 2.3194644

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542437, upper bound: 1.4523470
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552020, upper bound: 1.4526287
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6659822, 2.6690445
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6385803, 2.6345668
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6761675, 2.6609054
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7940507, 2.8030281
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5937948, 1.5851827
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6798077, 2.6842337
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9512348, 2.9545226
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2556753, 2.2599251
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8059378, 1.8148925
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3185987, 2.3206155

Time for backsubstitution: 14.58 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3557186126708984
rel_dist={9: [-1.46816030629895, 1.4681598898989336]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586212, upper bound: 1.3600267
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600268, upper bound: 1.3586184
time: 4.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 9, lower bound: -1.3586212, upper bound: 1.3600267
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.70
Output dim: 9, lower bound: -1.3600268, upper bound: 1.3586184

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5278611, 2.5297484
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5603666, 2.5612769
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5310321, 2.5365391
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7065969, 2.7026911
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5871203, 1.5888431
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5478172, 2.5408750
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8627281, 2.8635464
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1857963, 2.1878104
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7680435, 1.7652438
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2966604, 2.2977853

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3581335, upper bound: 1.3600220
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586149, upper bound: 1.3595388
time: 7.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5297494, 2.5278602
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5612774, 2.5603662
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5365386, 2.5310326
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7026906, 2.7065973
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5888431, 1.5871205
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5408745, 2.5478177
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8635464, 2.8627281
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1878104, 2.1857965
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7652435, 1.7680438
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2977853, 2.2966607

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3595388, upper bound: 1.3586139
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600228, upper bound: 1.3581334
time: 5.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 9, lower bound: -1.3581335, upper bound: 1.3600220
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 9, lower bound: -1.3586149, upper bound: 1.3595388
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 9, lower bound: -1.3595388, upper bound: 1.3586139
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.91
Output dim: 9, lower bound: -1.3600228, upper bound: 1.3581334

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5132971, 2.5095139
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5531940, 2.5513282
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5242143, 2.5316281
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7060933, 2.7023268
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5776546, 1.5756876
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5399418, 2.5352006
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8470469, 2.8417630
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1855645, 2.1874917
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7648935, 1.7608683
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2917056, 2.2942195

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3581162, upper bound: 1.3514023
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495100, upper bound: 1.3600088
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5076265, 2.5151849
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5504179, 2.5541043
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5261226, 2.5297208
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7062325, 2.7021871
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5739648, 1.5793765
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5421419, 2.5330000
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8409452, 2.8478651
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1854777, 2.1875784
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7636681, 1.7620931
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2930951, 2.2928305

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3509159
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3499936, upper bound: 1.3595238
time: 7.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5151854, 2.5076261
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5541048, 2.5504184
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5297208, 2.5261221
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7021871, 2.7062330
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5793769, 1.5739648
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5329990, 2.5421419
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8478651, 2.8409448
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1875787, 2.1854777
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7620935, 1.7636683
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2928305, 2.2930949

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3499934
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509164, upper bound: 1.3586001
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5095139, 2.5132971
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5513287, 2.5531945
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5316291, 2.5242143
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7023263, 2.7060933
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5756872, 1.5776541
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5352001, 2.5399423
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8417635, 2.8470464
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1874919, 2.1855645
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7608681, 1.7648938
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2942195, 2.2917056

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3495095
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3513999, upper bound: 1.3581163
time: 7.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3581162, upper bound: 1.3514023
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3495100, upper bound: 1.3600088
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3509159
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3499936, upper bound: 1.3595238
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3499934
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3509164, upper bound: 1.3586001
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3495124, upper bound: 1.3495095
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.14
Output dim: 9, lower bound: -1.3513999, upper bound: 1.3581163

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5155592, 2.5129147
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5209808, 2.5231423
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5366044, 2.5472693
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7258840, 2.7198887
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5244551, 1.5291517
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5491915, 2.5456238
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8292723, 2.8262110
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1714077, 2.1694040
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7493620, 1.7431183
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2695546, 2.2689047

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578342, upper bound: 1.3502310
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483463, upper bound: 1.3502406
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5166969, 2.5117769
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5250092, 2.5191145
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5398555, 2.5440183
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7236552, 2.7221169
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5311184, 1.5224884
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5503664, 2.5444503
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8314943, 2.8239889
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1674771, 2.1733348
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7471437, 1.7453361
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2663913, 2.2720680

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483479, upper bound: 1.3502372
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483403, upper bound: 1.3597247
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5110264, 2.5174475
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5222321, 2.5218906
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5417628, 2.5421109
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7237954, 2.7219772
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5274291, 1.5261772
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5525656, 2.5422492
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8253918, 2.8300910
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1673903, 2.1734216
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7459183, 1.7465611
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2677803, 2.2706788

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3488324, upper bound: 1.3497527
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3488248, upper bound: 1.3592390
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5185852, 2.5098886
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5259190, 2.5182042
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5453620, 2.5385122
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7197490, 2.7260232
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5328407, 1.5207655
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5434237, 2.5513916
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8323126, 2.8231707
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1694908, 2.1713209
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7443428, 1.7481363
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2675157, 2.2709432

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3497550, upper bound: 1.3488332
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497475, upper bound: 1.3583194
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5129147, 2.5155592
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5231428, 2.5209804
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5472693, 2.5366044
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7198892, 2.7258835
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5291519, 1.5244548
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5456228, 2.5491920
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8262110, 2.8292723
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1694040, 2.1714077
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7431183, 1.7493615
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2689052, 2.2695541

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502396, upper bound: 1.3483463
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483427, upper bound: 1.3578333
time: 5.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3578342, upper bound: 1.3502310
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3483463, upper bound: 1.3502406
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3483479, upper bound: 1.3502372
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3483403, upper bound: 1.3597247
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3488324, upper bound: 1.3497527
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3488248, upper bound: 1.3592390
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3497550, upper bound: 1.3488332
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3497475, upper bound: 1.3583194
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3502396, upper bound: 1.3483463
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 9, lower bound: -1.3483427, upper bound: 1.3578333

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5177155, 2.5154190
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5141926, 2.5172029
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5414362, 2.5528827
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7311912, 2.7244577
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5124164, 1.5188756
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5507336, 2.5474138
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8244953, 2.8220291
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1775160, 2.1743283
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7460508, 1.7393348
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2635136, 2.2620006

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578287, upper bound: 1.3502279
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3575360, upper bound: 1.3497426
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5192013, 2.5139322
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5190697, 2.5123262
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5454683, 2.5488501
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7282243, 2.7274246
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5208421, 1.5104499
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5521555, 2.5459914
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8273125, 2.8192105
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1724014, 2.1794429
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7433605, 1.7420256
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2594867, 2.2660272

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483349, upper bound: 1.3597212
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3592343
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5135307, 2.5196033
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5162926, 2.5151024
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5473766, 2.5469422
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7283635, 2.7272844
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5171528, 1.5141387
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5543556, 2.5437908
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8212109, 2.8253126
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1723146, 2.1795297
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7421350, 1.7432506
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2608757, 2.2646379

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483362, upper bound: 1.3590500
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3488210, upper bound: 1.3592336
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5210896, 2.5120444
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5199795, 2.5114160
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5509748, 2.5433440
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7243180, 2.7313309
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5225644, 1.5087271
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5452127, 2.5529327
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8281307, 2.8183923
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1744156, 2.1774290
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7405596, 1.7448258
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2606115, 2.2649024

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497420, upper bound: 1.3583158
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495581, upper bound: 1.3578292
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5154190, 2.5177155
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5172024, 2.5141921
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5528822, 2.5414357
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7244573, 2.7311907
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5188756, 1.5124164
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5474138, 2.5507331
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8220291, 2.8244944
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1743288, 2.1775157
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7393351, 1.7460511
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2620006, 2.2635133

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497432, upper bound: 1.3575358
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502282, upper bound: 1.3578279
time: 4.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3578287, upper bound: 1.3502279
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3575360, upper bound: 1.3497426
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3483349, upper bound: 1.3597212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3592343
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3483362, upper bound: 1.3590500
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3488210, upper bound: 1.3592336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3497420, upper bound: 1.3583158
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3495581, upper bound: 1.3578292
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3497432, upper bound: 1.3575358
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.37
Output dim: 9, lower bound: -1.3502282, upper bound: 1.3578279

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5161762, 2.5123749
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5136385, 2.5161090
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5385904, 2.5514483
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7311873, 2.7244501
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5116677, 1.5173905
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5499210, 2.5470033
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8226047, 2.8182940
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1772876, 2.1738768
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7457099, 1.7386613
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2621922, 2.2613320

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3346098, upper bound: 1.3329758
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3403446, upper bound: 1.3271831
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5146713, 2.5138793
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5130987, 2.5166407
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5400019, 2.5500369
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7311835, 2.7244463
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5109315, 1.5181267
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5503178, 2.5466022
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8207583, 2.8201385
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1770644, 2.1740973
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7453771, 1.7389920
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2628450, 2.2606795

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.41 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3343192, upper bound: 1.3321661
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3403387, upper bound: 1.3264804
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5176620, 2.5108886
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5185137, 2.5112324
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5426226, 2.5474157
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7282205, 2.7274170
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5200934, 1.5089648
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5513439, 2.5455809
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8254218, 2.8154755
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1721730, 2.1789913
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7430186, 1.7413521
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2581654, 2.2653587

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.41 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3251056, upper bound: 1.3425085
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3308124, upper bound: 1.3366792
time: 8.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5161572, 2.5123925
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5179758, 2.5117640
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5440340, 2.5460043
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7282166, 2.7274132
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5193572, 1.5097008
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5517406, 2.5451794
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8235774, 2.8173203
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1719499, 2.1792119
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7426867, 1.7416828
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2588181, 2.2647061

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3248207, upper bound: 1.3416979
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3308086, upper bound: 1.3359842
time: 5.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5119915, 2.5165596
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5157309, 2.5140085
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5445309, 2.5455079
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7283521, 2.7272768
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5164037, 1.5126536
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5535431, 2.5433750
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8193202, 2.8215775
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1720834, 2.1790781
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7417922, 1.7425768
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2595549, 2.2639694

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3251076, upper bound: 1.3416905
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3308141, upper bound: 1.3357782
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5104866, 2.5180635
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5151987, 2.5145478
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5459423, 2.5440965
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7283559, 2.7272811
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5156674, 1.5133898
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5539446, 2.5429788
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8174758, 2.8234224
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1718631, 2.1793013
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7414613, 1.7429092
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2602072, 2.2633169

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3258104, upper bound: 1.3416982
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3316215, upper bound: 1.3359817
time: 10.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5195503, 2.5090003
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5194254, 2.5103221
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5481291, 2.5419097
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7243142, 2.7313232
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5218158, 1.5072420
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5444012, 2.5525222
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8262401, 2.8146567
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1741872, 2.1769774
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7402186, 1.7441523
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2592902, 2.2642341

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3264783, upper bound: 1.3411534
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3321650, upper bound: 1.3353071
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5180454, 2.5105047
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5188856, 2.5108538
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5495405, 2.5404983
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7243104, 2.7313194
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5210795, 1.5079782
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5447979, 2.5521212
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8243957, 2.8165021
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1739640, 2.1771979
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7398858, 1.7444828
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2599430, 2.2635813

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3262789, upper bound: 1.3403462
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3321584, upper bound: 1.3346118
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5138798, 2.5146718
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5166407, 2.5130982
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5500364, 2.5400019
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7244458, 2.7311831
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5181270, 1.5109313
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5466022, 2.5503178
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8201385, 2.8207593
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1740975, 2.1770642
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7389922, 1.7453775
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2606792, 2.2628448

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3264803, upper bound: 1.3403382
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3321666, upper bound: 1.3343216
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5123749, 2.5161757
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5161085, 2.5136371
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5514479, 2.5385900
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7244496, 2.7311873
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5173907, 1.5116675
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5470028, 2.5499215
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8182940, 2.8226037
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1738772, 2.1772873
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7386613, 1.7457099
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2613320, 2.2621922

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 909

Time for candidate selection: 0.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3271830, upper bound: 1.3403443
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3329763, upper bound: 1.3346097
time: 4.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3346098, upper bound: 1.3329758
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3403446, upper bound: 1.3271831
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3343192, upper bound: 1.3321661
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3403387, upper bound: 1.3264804
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3251056, upper bound: 1.3425085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3308124, upper bound: 1.3366792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3248207, upper bound: 1.3416979
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3308086, upper bound: 1.3359842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3251076, upper bound: 1.3416905
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3308141, upper bound: 1.3357782
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3258104, upper bound: 1.3416982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3316215, upper bound: 1.3359817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3264783, upper bound: 1.3411534
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3321650, upper bound: 1.3353071
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3262789, upper bound: 1.3403462
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3321584, upper bound: 1.3346118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3264803, upper bound: 1.3403382
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3321666, upper bound: 1.3343216
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3271830, upper bound: 1.3403443
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.13
Output dim: 9, lower bound: -1.3329763, upper bound: 1.3346097
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2352.89 seconds
