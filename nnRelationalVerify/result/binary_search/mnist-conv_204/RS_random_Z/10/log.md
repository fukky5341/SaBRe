## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.67154946715
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8134632, -6.7279902, -8.8134632, -6.7279902, -2.0854731, 2.0854731)
1: (1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.6906104, 1.6906104)
2: (-5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.5822163, 1.5822163)
3: (-10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.5740218, 1.5740218)
4: (-4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704)
5: (-8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.5837355, 1.5837355)
6: (-5.9832397, -3.9410968, -5.9832397, -3.9410968, -2.0421429, 2.0421429)
7: (-4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261)
8: (-3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.4418497, 1.4418497)
9: (-11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.9131136, 1.9131136)

## BASE Result
execution time: IAR + LP analysis = 15.15 + 31.73 = 46.89 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.11 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.4624950885772705
rel_dist={1: [-0.9108364634088741, 0.9108356417165169]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.2293808460235596
rel_dist={1: [-0.49714962017070174, 0.49714935310829933]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2760038375854492
rel_dist={1: [-0.5884148639986186, 0.5884165441500335]}

## Binary Search Result
Binary search time: 195.09 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3358.02 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720632, upper bound: 0.9849544
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849542, upper bound: 0.9720634
time: 3.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.29
Output dim: 1, lower bound: -0.9720632, upper bound: 0.9849544
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.29
Output dim: 1, lower bound: -0.9849542, upper bound: 0.9720634

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8403149, 1.8168218
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5073885, 1.5078933
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3702402, 1.3847668
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2166235, 1.2224830
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3323188, 1.3075553
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8274641, 1.8171057
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2983720, 1.2921588
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6230278, 1.6082753

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684903, upper bound: 0.9849483
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720574, upper bound: 0.9813807
time: 3.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8168216, 1.8403149
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5078932, 1.5073886
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3847668, 1.3702402
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2224829, 1.2166237
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3075552, 1.3323189
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8171058, 1.8274642
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2921588, 1.2983720
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6082754, 1.6230274

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9818386, upper bound: 0.9714209
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9843119, upper bound: 0.9689474
time: 4.05 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9684903, upper bound: 0.9849483
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9720574, upper bound: 0.9813807
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9818386, upper bound: 0.9714209
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9843119, upper bound: 0.9689474

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8390265, 1.8131828
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5037395, 1.5066037
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3598793, 1.3811166
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2159913, 1.2207035
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3306215, 1.3027487
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8233731, 1.8055466
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2941775, 1.2802966
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6208115, 1.6020119

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9651417, upper bound: 0.9849375
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684751, upper bound: 0.9816164
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8366756, 1.8155332
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5060989, 1.5042441
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3665900, 1.3744059
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2148440, 1.2218505
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3275123, 1.3058579
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8159053, 1.8130141
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2865095, 1.2879645
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6167641, 1.6060592

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720557, upper bound: 0.9650252
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556966, upper bound: 0.9813787
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8169811, 1.8397849
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5071656, 1.5076078
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3833513, 1.3706698
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2227044, 1.2158798
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3076808, 1.3319067
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8174551, 1.8262945
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2923594, 1.2977070
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6087093, 1.6215798

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9817999, upper bound: 0.9704340
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9808597, upper bound: 0.9713823
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8162916, 1.8403149
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5078932, 1.5066611
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3847668, 1.3688250
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2217393, 1.2166237
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3071432, 1.3323189
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8159359, 1.8274642
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2914939, 1.2983720
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6068277, 1.6230274

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9813732, upper bound: 0.9684812
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9843060, upper bound: 0.9684805
time: 4.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9651417, upper bound: 0.9849375
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9684751, upper bound: 0.9816164
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9720557, upper bound: 0.9650252
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9556966, upper bound: 0.9813787
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9817999, upper bound: 0.9704340
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9808597, upper bound: 0.9713823
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9813732, upper bound: 0.9684812
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.40
Output dim: 1, lower bound: -0.9843060, upper bound: 0.9684805

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8393342, 1.8124974
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5025188, 1.5071257
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3583052, 1.3817990
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2170570, 1.2182015
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3299313, 1.3030246
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8244686, 1.8029639
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2930570, 1.2807885
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6214118, 1.6005886

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9560558, upper bound: 0.9849017
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9651050, upper bound: 0.9758284
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8383405, 1.8131828
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5037395, 1.5053829
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3598793, 1.3795426
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2134893, 1.2207035
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3306215, 1.3020583
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8207903, 1.8055466
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2941775, 1.2791761
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6193881, 1.6020119

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684729, upper bound: 0.9658077
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9526672, upper bound: 0.9816145
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8957119, 1.8602281
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5248426, 1.5290086
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3700819, 1.3790185
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2544725, 1.2505116
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3200328, 1.3005604
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8352990, 1.8386333
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2882777, 1.2903010
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6238482, 1.6114321

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720170, upper bound: 0.9640449
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9710686, upper bound: 0.9649864
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8813705, 1.8745689
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5308632, 1.5229876
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3712025, 1.3778977
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2435050, 1.2614782
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3222148, 1.2983782
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8415241, 1.8324082
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2888461, 1.2897326
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6221373, 1.6131430

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9487861, upper bound: 0.9813631
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556855, upper bound: 0.9780285
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7967987, 1.8112974
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5125782, 1.5117345
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3824043, 1.3699994
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2209214, 1.2135769
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2986255, 1.3254902
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7919936, 1.8082614
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2925035, 1.2978958
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6111774, 1.6248169

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9817978, upper bound: 0.9546220
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9659892, upper bound: 0.9704316
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7884932, 1.8196192
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5112922, 1.5130200
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3826809, 1.3697228
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2204021, 1.2140968
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3012686, 1.3228514
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7994285, 1.8008327
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2925483, 1.2978510
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6119461, 1.6240473

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9808565, upper bound: 0.9550234
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9645007, upper bound: 0.9713802
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8150029, 1.8366759
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5042440, 1.5053716
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3744056, 1.3651748
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2211065, 1.2148441
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3054452, 1.3275123
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8118448, 1.8159052
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2873040, 1.2865099
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6046109, 1.6167641

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9813345, upper bound: 0.9674940
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9803930, upper bound: 0.9684423
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8126531, 1.8390262
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5066036, 1.5030117
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3811164, 1.3584640
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2199593, 1.2159911
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3023362, 1.3306215
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8043776, 1.8233730
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2796360, 1.2941777
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6005640, 1.6208113

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9752039, upper bound: 0.9684457
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9842704, upper bound: 0.9593948
time: 4.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9560558, upper bound: 0.9849017
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9651050, upper bound: 0.9758284
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9684729, upper bound: 0.9658077
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9526672, upper bound: 0.9816145
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9720170, upper bound: 0.9640449
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9710686, upper bound: 0.9649864
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9487861, upper bound: 0.9813631
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9556855, upper bound: 0.9780285
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9817978, upper bound: 0.9546220
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9659892, upper bound: 0.9704316
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9808565, upper bound: 0.9550234
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9645007, upper bound: 0.9713802
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9813345, upper bound: 0.9674940
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9803930, upper bound: 0.9684423
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9752039, upper bound: 0.9684457
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -0.9842704, upper bound: 0.9593948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8511021, 1.7993290
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.4966923, 1.5123351
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3504748, 1.3887978
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2141051, 1.2208482
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3339581, 1.2985185
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8239846, 1.8033954
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2969232, 1.2764785
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6396518, 1.5802045

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9560157, upper bound: 0.9839225
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9550676, upper bound: 0.9848629
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8261654, 1.8124974
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5025188, 1.5012991
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3583052, 1.3739686
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2170570, 1.2152494
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3254251, 1.3030246
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8244686, 1.8024796
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2887468, 1.2807885
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6010275, 1.6005886

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9651029, upper bound: 0.9594712
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9487529, upper bound: 0.9758264
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8384030, 1.8132262
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5037603, 1.5054131
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3598847, 1.3795500
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2134787, 1.2206886
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3306420, 1.3020729
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8208342, 1.8056101
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2941607, 1.2791529
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6193950, 1.6020240

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684636, upper bound: 0.9651463
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684655, upper bound: 0.9622120
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8383844, 1.8132446
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5037694, 1.5054038
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3598869, 1.3795478
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2134744, 1.2206931
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3306360, 1.3020788
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8208532, 1.8055912
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2941543, 1.2791593
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6193998, 1.6020188

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9526643, upper bound: 0.9652601
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9487845, upper bound: 0.9658057
time: 6.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8755460, 1.8317404
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5302551, 1.5331353
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3691342, 1.3783474
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2526891, 1.2482090
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3109777, 1.2941483
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8098373, 1.8206065
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2884219, 1.2904897
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6263154, 1.6146691

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686855, upper bound: 0.9640298
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8672242, 1.8400459
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5289693, 1.5344211
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3694108, 1.3780708
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2521694, 1.2487283
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3136165, 1.2915052
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8172660, 1.8131716
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2884667, 1.2904449
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6270850, 1.6138998

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9710665, upper bound: 0.9649863
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9552551, upper bound: 0.9649857
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8816786, 1.8738837
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5296429, 1.5235097
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3696284, 1.3785801
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2445712, 1.2589767
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3215244, 1.2986542
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8426194, 1.8298247
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2877258, 1.2902247
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6227374, 1.6117196

Time for backsubstitution: 16.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803825
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8806858, 1.8745689
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5308632, 1.5217671
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3712025, 1.3763237
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2410033, 1.2614782
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3222148, 1.2976879
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8389406, 1.8324082
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2888461, 1.2886122
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6207137, 1.6131430

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556839, upper bound: 0.9622195
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556854, upper bound: 0.9780264
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7968607, 1.8113410
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5125991, 1.5117644
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3824093, 1.3700066
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2209108, 1.2135620
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2986465, 1.3255051
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7920380, 1.8083247
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2924861, 1.2978722
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6111846, 1.6248293

Time for backsubstitution: 17.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9726993, upper bound: 0.9545867
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9817625, upper bound: 0.9455169
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7968426, 1.8113594
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5126082, 1.5117553
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3824115, 1.3700047
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2209063, 1.2135663
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2986405, 1.3255112
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7920570, 1.8083059
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2924799, 1.2978786
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6111894, 1.6248243

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9655235, upper bound: 0.9704257
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9655239, upper bound: 0.9674933
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8475289, 1.8643143
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5300362, 1.5377846
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3861721, 1.3743348
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2600296, 1.2427578
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2937894, 1.3175540
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8188217, 1.8264511
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2943164, 1.3001875
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6190298, 1.6294203

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9803893, upper bound: 0.9550177
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9803896, upper bound: 0.9520887
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8331885, 1.8786552
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5360570, 1.5317640
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3872926, 1.3732140
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2490630, 1.2537251
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2959714, 1.3153719
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8250468, 1.8202260
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2948848, 1.2996191
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6173189, 1.6311312

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9554049, upper bound: 0.9713449
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9644694, upper bound: 0.9622780
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7948213, 1.8081887
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5096562, 1.5094981
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3734589, 1.3645043
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2193234, 1.2125409
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2963903, 1.3210957
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7863841, 1.7978725
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2874477, 1.2866987
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6070786, 1.6200013

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9813324, upper bound: 0.9516849
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9655255, upper bound: 0.9674918
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7865157, 1.8165104
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5083706, 1.5107839
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3737354, 1.3642280
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2188041, 1.2130607
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2990334, 1.3184569
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7938190, 1.7904439
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2874925, 1.2866539
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6078482, 1.6192318

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9803908, upper bound: 0.9526333
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9645839, upper bound: 0.9684403
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8244205, 1.8258574
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5007764, 1.5082211
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3732862, 1.3654656
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2170074, 1.2186365
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3063598, 1.3261154
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8038931, 1.8238045
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2835019, 1.2898675
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6188011, 1.6004276

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9751652, upper bound: 0.9674590
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9742231, upper bound: 0.9684071
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7994843, 1.8390262
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5066036, 1.4971852
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3811164, 1.3506341
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2199593, 1.2130390
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2978299, 1.3306215
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8043776, 1.8228887
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2753255, 1.2941777
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.5801802, 1.6208113

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9842683, upper bound: 0.9435860
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684587, upper bound: 0.9593926
time: 4.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9560157, upper bound: 0.9839225
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9550676, upper bound: 0.9848629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9651029, upper bound: 0.9594712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9487529, upper bound: 0.9758264
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9684636, upper bound: 0.9651463
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9684655, upper bound: 0.9622120
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9526643, upper bound: 0.9652601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9487845, upper bound: 0.9658057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9686855, upper bound: 0.9640298
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9710665, upper bound: 0.9649863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9552551, upper bound: 0.9649857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803825
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9556839, upper bound: 0.9622195
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9556854, upper bound: 0.9780264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9726993, upper bound: 0.9545867
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9817625, upper bound: 0.9455169
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9655235, upper bound: 0.9704257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9655239, upper bound: 0.9674933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9803893, upper bound: 0.9550177
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9803896, upper bound: 0.9520887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9554049, upper bound: 0.9713449
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9644694, upper bound: 0.9622780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9813324, upper bound: 0.9516849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9655255, upper bound: 0.9674918
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9803908, upper bound: 0.9526333
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9645839, upper bound: 0.9684403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9751652, upper bound: 0.9674590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9742231, upper bound: 0.9684071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9842683, upper bound: 0.9435860
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.84
Output dim: 1, lower bound: -0.9684587, upper bound: 0.9593926

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8309364, 1.7708416
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5021045, 1.5164616
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3495283, 1.3881278
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2123222, 1.2185464
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3249028, 1.2921065
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.7985225, 1.7853683
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2970670, 1.2766676
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6421189, 1.5834415

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9560064, upper bound: 0.9832778
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9560083, upper bound: 0.9803412
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8226147, 1.7791471
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5008187, 1.5177462
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3498046, 1.3878510
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2118025, 1.2190655
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3275397, 1.2894633
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8059506, 1.7779332
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2971123, 1.2766228
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6428885, 1.5826721

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9550570, upper bound: 0.9842187
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9550589, upper bound: 0.9812824
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8852019, 1.8571925
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5212625, 1.5260634
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3617971, 1.3785813
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2566857, 1.2439189
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3179455, 1.2977272
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8438621, 1.8280975
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2905158, 1.2831253
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6081119, 1.6059616

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9650936, upper bound: 0.9588355
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9650955, upper bound: 0.9559196
time: 3.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9560064, upper bound: 0.9832778
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9560083, upper bound: 0.9803412
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9550570, upper bound: 0.9842187
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9550589, upper bound: 0.9812824
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9650936, upper bound: 0.9588355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.05
Output dim: 1, lower bound: -0.9650955, upper bound: 0.9559196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9487529, upper bound: 0.9758264
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9684636, upper bound: 0.9651463
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9684655, upper bound: 0.9622120
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9526643, upper bound: 0.9652601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9487845, upper bound: 0.9658057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9686855, upper bound: 0.9640298
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9710665, upper bound: 0.9649863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9552551, upper bound: 0.9649857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803825
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9556839, upper bound: 0.9622195
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9556854, upper bound: 0.9780264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9726993, upper bound: 0.9545867
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9817625, upper bound: 0.9455169
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9655235, upper bound: 0.9704257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9655239, upper bound: 0.9674933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9803893, upper bound: 0.9550177
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9803896, upper bound: 0.9520887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9554049, upper bound: 0.9713449
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9644694, upper bound: 0.9622780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9813324, upper bound: 0.9516849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9655255, upper bound: 0.9674918
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9803908, upper bound: 0.9526333
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9645839, upper bound: 0.9684403
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9751652, upper bound: 0.9674590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9742231, upper bound: 0.9684071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9842683, upper bound: 0.9435860
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.05
Output dim: 1, lower bound: -0.9684587, upper bound: 0.9593926
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.509117841720581
rel_dist={1: [-0.9849775715086642, 0.9849766529340247]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555036, upper bound: 0.7549682
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549682, upper bound: 0.7555036
time: 3.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 1, lower bound: -0.7555036, upper bound: 0.7549682
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.39
Output dim: 1, lower bound: -0.7549682, upper bound: 0.7555036

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6910863, 1.6863406
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3741107, 1.3733760
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3216949, 1.3218529
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0326273, 1.0323303
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4088378, 1.4092615
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2126222, 1.2141327
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6359453, 1.6401938
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3624816, 1.3626409
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1603085, 1.1603341
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4818325, 1.4822719

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7535360, upper bound: 0.7549651
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555006, upper bound: 0.7530006
time: 3.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6863408, 1.6910865
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3733759, 1.3741108
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3218529, 1.3216949
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0323303, 1.0326273
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4092612, 1.4088376
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2141328, 1.2126223
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6401939, 1.6359454
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3626409, 1.3624816
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1603340, 1.1603085
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4822721, 1.4818323

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474841, upper bound: 0.7554915
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549566, upper bound: 0.7480205
time: 3.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 1, lower bound: -0.7535360, upper bound: 0.7549651
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 1, lower bound: -0.7555006, upper bound: 0.7530006
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 1, lower bound: -0.7474841, upper bound: 0.7554915
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 1, lower bound: -0.7549566, upper bound: 0.7480205

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6909499, 1.6858106
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3733833, 1.3731894
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3202798, 1.3214922
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0324352, 1.0315866
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4079633, 1.4090378
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2125170, 1.2137201
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6356444, 1.6390249
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3611379, 1.3623002
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1601379, 1.1596687
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4814606, 1.4808247

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7460528, upper bound: 0.7549535
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7535231, upper bound: 0.7474794
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6905565, 1.6862042
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3739243, 1.3726487
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3213341, 1.3204379
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0318837, 1.0321381
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4086142, 1.4083867
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2122097, 1.2140274
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6347761, 1.6398929
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3621402, 1.3612976
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1596432, 1.1601634
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4803858, 1.4819002

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529278, upper bound: 0.7529980
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554981, upper bound: 0.7504278
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6191852, 1.6105063
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3716464, 1.3726693
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2721107, 1.2802532
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0121891, 1.0158345
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3395936, 1.3508012
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1434023, 1.1277437
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6294308, 1.6192665
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3893359, 1.3936636
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1425271, 1.1389511
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4399824, 1.4311123

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455148, upper bound: 0.7554876
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474794, upper bound: 0.7535232
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6057603, 1.6239402
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3719344, 1.3723811
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2804115, 1.2719524
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0155375, 1.0124862
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3512256, 1.3391697
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1292541, 1.1418943
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6235151, 1.6251860
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3938239, 1.3891761
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1389766, 1.1425016
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4315519, 1.4395421

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529890, upper bound: 0.7480174
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549536, upper bound: 0.7460529
time: 3.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7460528, upper bound: 0.7549535
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7535231, upper bound: 0.7474794
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7529278, upper bound: 0.7529980
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7554981, upper bound: 0.7504278
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7455148, upper bound: 0.7554876
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7474794, upper bound: 0.7535232
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7529890, upper bound: 0.7480174
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.98
Output dim: 1, lower bound: -0.7549536, upper bound: 0.7460529

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6238036, 1.6052299
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3716540, 1.3717484
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2705376, 1.2800508
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0122942, 1.0147940
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3382950, 1.3510020
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1417892, 1.1288416
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6248846, 1.6223454
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3878319, 1.3934820
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1423310, 1.1383116
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4391699, 1.4301043

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7434800, upper bound: 0.7549511
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7460503, upper bound: 0.7523808
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6103697, 1.6186545
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3719423, 1.3714600
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2788384, 1.2717500
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0156425, 1.0114455
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3499265, 1.3393703
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1276383, 1.1429898
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6189656, 1.6282611
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3923199, 1.3889940
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1387807, 1.1418620
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4307404, 1.4385343

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530408, upper bound: 0.7474759
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530411, upper bound: 0.7450292
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6904392, 1.6855192
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3727038, 1.3724240
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3197606, 1.3201535
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0314205, 1.0296364
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4085469, 1.4080381
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2115195, 1.2138894
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6342950, 1.6373094
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3608961, 1.3610668
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1585226, 1.1599643
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4801178, 1.4804764

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529265, upper bound: 0.7440507
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7439541, upper bound: 0.7529966
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6898718, 1.6860867
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3736999, 1.3714281
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3210499, 1.3188643
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0293818, 1.0316751
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4082656, 1.4083197
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2120717, 1.2133372
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6321926, 1.6394116
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3619094, 1.3600531
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1594441, 1.1590430
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4789615, 1.4816327

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554968, upper bound: 0.7414677
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7465412, upper bound: 0.7504273
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6190486, 1.6099758
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3709192, 1.3724833
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2706957, 1.2798927
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0119971, 1.0150907
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3387189, 1.3505776
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1432970, 1.1273313
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6291294, 1.6180971
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3879912, 1.3933220
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1423568, 1.1382860
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4396095, 1.4296649

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455136, upper bound: 0.7465308
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7365983, upper bound: 0.7554875
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6186547, 1.6103697
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3714602, 1.3719423
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2717500, 1.2788384
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0114454, 1.0156424
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3393700, 1.3499265
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1429896, 1.1276386
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6282611, 1.6189651
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3889935, 1.3923197
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1418620, 1.1387807
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4385347, 1.4307401

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7449066, upper bound: 0.7535207
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474769, upper bound: 0.7509503
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6056237, 1.6234097
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3712075, 1.3721948
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2789965, 1.2715919
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0153455, 1.0117426
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3503509, 1.3389461
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1291490, 1.1414819
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6232138, 1.6240163
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3924792, 1.3888342
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1388062, 1.1418364
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4311800, 1.4380946

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7504162, upper bound: 0.7480149
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529865, upper bound: 0.7454447
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6052299, 1.6238036
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3717484, 1.3716540
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2800508, 1.2705376
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0147940, 1.0122942
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3510020, 1.3382950
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1288416, 1.1417892
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6223454, 1.6248846
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3934815, 1.3878319
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1383115, 1.1423311
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4301043, 1.4391699

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7525078, upper bound: 0.7455708
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549484, upper bound: 0.7455706
time: 3.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7434800, upper bound: 0.7549511
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7460503, upper bound: 0.7523808
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7530408, upper bound: 0.7474759
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7530411, upper bound: 0.7450292
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7529265, upper bound: 0.7440507
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7439541, upper bound: 0.7529966
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7554968, upper bound: 0.7414677
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7465412, upper bound: 0.7504273
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7455136, upper bound: 0.7465308
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7365983, upper bound: 0.7554875
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7449066, upper bound: 0.7535207
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7474769, upper bound: 0.7509503
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7504162, upper bound: 0.7480149
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7529865, upper bound: 0.7454447
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7525078, upper bound: 0.7455708
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.16
Output dim: 1, lower bound: -0.7549484, upper bound: 0.7455706

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6236858, 1.6045449
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3704336, 1.3715236
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2689641, 1.2797666
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0118310, 1.0122923
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3382282, 1.3506536
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1410990, 1.1287035
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6244035, 1.6197619
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3865883, 1.3932517
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1412109, 1.1381129
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4389029, 1.4286809

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7424595, upper bound: 0.7549449
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7429980, upper bound: 0.7525052
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6231184, 1.6051123
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3714292, 1.3705277
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2702534, 1.2784772
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0097923, 1.0143311
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3379467, 1.3509355
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1416512, 1.1281513
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6223011, 1.6218643
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3876021, 1.3922379
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1421322, 1.1371914
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4377465, 1.4298372

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7460491, upper bound: 0.7434149
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7371273, upper bound: 0.7523803
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6080737, 1.6150157
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3682930, 1.3691595
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2684777, 1.2652254
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0145197, 1.0096657
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3418667, 1.3342891
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1246085, 1.1381831
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6116743, 1.6167030
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3812201, 1.3819995
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1313045, 1.1300043
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4267931, 1.4322708

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530396, upper bound: 0.7385515
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7441353, upper bound: 0.7474731
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6067309, 1.6163588
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3696413, 1.3678110
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2723124, 1.2613893
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0138624, 1.0103213
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3448455, 1.3313103
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1228318, 1.1399598
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6074076, 1.6209702
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3853257, 1.3778946
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1269228, 1.1343858
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4244766, 1.4345834

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530382, upper bound: 0.7357614
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7437681, upper bound: 0.7450269
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6904931, 1.6855624
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3727248, 1.3724501
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3197653, 1.3201594
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0314080, 1.0296212
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4085126, 1.4080093
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2115374, 1.2139039
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6343398, 1.6373652
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3609183, 1.3610852
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1585026, 1.1599410
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4801273, 1.4804883

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7439358, upper bound: 0.7440206
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528971, upper bound: 0.7350367
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6904826, 1.6855729
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3727300, 1.3724449
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3197665, 1.3201582
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0314053, 1.0296237
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4085183, 1.4080033
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2115340, 1.2139072
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6343508, 1.6373545
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3609145, 1.3610895
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1584992, 1.1599447
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4801302, 1.4804852

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7439520, upper bound: 0.7437251
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436415, upper bound: 0.7529939
time: 4.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6899257, 1.6861298
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3737206, 1.3714545
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3210547, 1.3188701
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0293692, 1.0316601
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4082308, 1.4082909
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2120895, 1.2133517
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6322374, 1.6394674
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3619320, 1.3600714
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1594243, 1.1590195
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4789705, 1.4816446

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7480137, upper bound: 0.7414563
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554839, upper bound: 0.7340115
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6899152, 1.6861403
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3737259, 1.3714492
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3210559, 1.3188689
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0293666, 1.0316626
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4082370, 1.4082849
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2120862, 1.2133551
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6322484, 1.6394564
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3619282, 1.3600757
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1594205, 1.1590232
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4789734, 1.4816418

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7441467, upper bound: 0.7499452
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7465360, upper bound: 0.7499449
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6191025, 1.6100197
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3709403, 1.3725094
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2707007, 1.2798991
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0119846, 1.0150759
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3386850, 1.3505497
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1433156, 1.1273463
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6291738, 1.6181523
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3880148, 1.3933406
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1423367, 1.1382624
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4396195, 1.4296772

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455107, upper bound: 0.7462153
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7362409, upper bound: 0.7465234
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6190920, 1.6100302
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3709456, 1.3725041
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2707019, 1.2798977
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0119820, 1.0150783
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3386910, 1.3505437
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1433120, 1.1273496
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6291847, 1.6181415
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3880100, 1.3933451
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1423329, 1.1382661
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4396224, 1.4296741

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7340113, upper bound: 0.7554847
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7365942, upper bound: 0.7529145
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6185369, 1.6096847
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3702397, 1.3717175
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2701764, 1.2785542
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0109825, 1.0131406
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3393035, 1.3495781
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1422994, 1.1275004
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6277800, 1.6163816
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3877499, 1.3920894
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1407417, 1.1385820
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4382672, 1.4293168

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7449038, upper bound: 0.7442485
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7356336, upper bound: 0.7535186
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6179695, 1.6102521
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3712356, 1.3707218
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2714658, 1.2772648
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0089438, 1.0151795
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3390217, 1.3498597
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1428516, 1.1269481
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6256776, 1.6184838
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3887637, 1.3910758
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1416632, 1.1376605
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4371109, 1.4304731

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474741, upper bound: 0.7416723
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7382039, upper bound: 0.7509483
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6055059, 1.6227248
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3699870, 1.3719702
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2774229, 1.2713077
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0148826, 1.0092409
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3502841, 1.3385978
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1284585, 1.1413437
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6227326, 1.6214328
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3912356, 1.3886039
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1376861, 1.1416377
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4309130, 1.4366713

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7414446, upper bound: 0.7479850
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7503871, upper bound: 0.7391022
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6049385, 1.6232922
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3709829, 1.3709743
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2787123, 1.2700183
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0128438, 1.0112797
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3500025, 1.3388796
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1290107, 1.1407915
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6206303, 1.6235349
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3922493, 1.3875904
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1386074, 1.1407163
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4297562, 1.4378276

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7499355, upper bound: 0.7454410
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7525044, upper bound: 0.7429988
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6029339, 1.6201649
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3680992, 1.3693531
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2696900, 1.2640116
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0136697, 1.0105144
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3429422, 1.3332140
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1258118, 1.1369826
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6150551, 1.6133265
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3823826, 1.3808384
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1308355, 1.1304734
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4261532, 1.4329064

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436449, upper bound: 0.7455407
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7524784, upper bound: 0.7367220
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6015911, 1.6215079
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3694475, 1.3680048
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2735262, 1.2601771
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0130141, 1.0111715
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3459210, 1.3302352
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1240351, 1.1387593
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6107874, 1.6175938
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3864872, 1.3767328
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1264539, 1.1348550
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4238405, 1.4352227

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7523756, upper bound: 0.7455680
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549459, upper bound: 0.7429978
time: 3.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7424595, upper bound: 0.7549449
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7429980, upper bound: 0.7525052
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7460491, upper bound: 0.7434149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7371273, upper bound: 0.7523803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7530396, upper bound: 0.7385515
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7441353, upper bound: 0.7474731
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7530382, upper bound: 0.7357614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7437681, upper bound: 0.7450269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7439358, upper bound: 0.7440206
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7528971, upper bound: 0.7350367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7439520, upper bound: 0.7437251
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7436415, upper bound: 0.7529939
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7480137, upper bound: 0.7414563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7554839, upper bound: 0.7340115
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7441467, upper bound: 0.7499452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7465360, upper bound: 0.7499449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7455107, upper bound: 0.7462153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7362409, upper bound: 0.7465234
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7340113, upper bound: 0.7554847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7365942, upper bound: 0.7529145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7449038, upper bound: 0.7442485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7356336, upper bound: 0.7535186
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7474741, upper bound: 0.7416723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7382039, upper bound: 0.7509483
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7414446, upper bound: 0.7479850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7503871, upper bound: 0.7391022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7499355, upper bound: 0.7454410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7525044, upper bound: 0.7429988
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7436449, upper bound: 0.7455407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7524784, upper bound: 0.7367220
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7523756, upper bound: 0.7455680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.16
Output dim: 1, lower bound: -0.7549459, upper bound: 0.7429978

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6213894, 1.6009054
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3667843, 1.3692229
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2586033, 1.2732420
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0107086, 1.0105126
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3301685, 1.3455727
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1380689, 1.1238968
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6171119, 1.6082034
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3754885, 1.3862567
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1337347, 1.1262552
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4349561, 1.4224173

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7341442, upper bound: 0.7549184
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7429677, upper bound: 0.7459881
time: 3.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6200466, 1.6022484
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3681326, 1.3678746
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2624381, 1.2694058
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0100515, 1.0111681
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3331473, 1.3425939
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1362922, 1.1256735
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6128442, 1.6124706
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3795941, 1.3821518
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1293530, 1.1306367
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4326396, 1.4247301

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7429951, upper bound: 0.7432331
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7337250, upper bound: 0.7525032
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6231728, 1.6051557
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3714501, 1.3705537
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2702582, 1.2784832
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0097799, 1.0143163
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3379121, 1.3509066
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1416693, 1.1281661
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6223454, 1.6219198
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3876238, 1.3922558
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1421126, 1.1371680
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4377561, 1.4298496

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455668, upper bound: 0.7434112
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455670, upper bound: 0.7410217
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6231623, 1.6051662
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3714553, 1.3705485
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2702594, 1.2784822
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0097775, 1.0143187
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3379180, 1.3509007
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1416657, 1.1281695
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6223564, 1.6219088
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3876195, 1.3922601
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1421088, 1.1371717
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4377589, 1.4298465

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7370861, upper bound: 0.7431028
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7367773, upper bound: 0.7523775
time: 3.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7341442, upper bound: 0.7549184
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7429677, upper bound: 0.7459881
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7429951, upper bound: 0.7432331
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7337250, upper bound: 0.7525032
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7455668, upper bound: 0.7434112
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7455670, upper bound: 0.7410217
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7370861, upper bound: 0.7431028
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.00
Output dim: 1, lower bound: -0.7367773, upper bound: 0.7523775
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7530396, upper bound: 0.7385515
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7441353, upper bound: 0.7474731
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7530382, upper bound: 0.7357614
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7437681, upper bound: 0.7450269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7439358, upper bound: 0.7440206
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7528971, upper bound: 0.7350367
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7439520, upper bound: 0.7437251
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7436415, upper bound: 0.7529939
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7480137, upper bound: 0.7414563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7554839, upper bound: 0.7340115
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7441467, upper bound: 0.7499452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7465360, upper bound: 0.7499449
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7455107, upper bound: 0.7462153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7362409, upper bound: 0.7465234
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7340113, upper bound: 0.7554847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7365942, upper bound: 0.7529145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7449038, upper bound: 0.7442485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7356336, upper bound: 0.7535186
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7474741, upper bound: 0.7416723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7382039, upper bound: 0.7509483
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7414446, upper bound: 0.7479850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7503871, upper bound: 0.7391022
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7499355, upper bound: 0.7454410
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7525044, upper bound: 0.7429988
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7436449, upper bound: 0.7455407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7524784, upper bound: 0.7367220
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7523756, upper bound: 0.7455680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.00
Output dim: 1, lower bound: -0.7549459, upper bound: 0.7429978
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3692495822906494
rel_dist={1: [-0.7555267226325175, 0.7555256824713692]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719671, upper bound: 0.6734570
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734558, upper bound: 0.6719684
time: 3.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 1, lower bound: -0.6719671, upper bound: 0.6734570
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.76
Output dim: 1, lower bound: -0.6734558, upper bound: 0.6719684

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6537521, 1.6534567
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3218994, 1.3223050
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2887805, 1.2895715
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9666290, 0.9662156
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3576009, 1.3580894
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1645689, 1.1643384
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6034160, 1.6027648
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3301785, 1.3309305
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1087527, 1.1083814
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4188457, 1.4180393

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719625, upper bound: 0.6665500
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6650619, upper bound: 0.6734523
time: 3.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6534569, 1.6537521
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3223052, 1.3218994
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2895713, 1.2887807
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9662154, 0.9666293
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3580894, 1.3576009
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1643384, 1.1645689
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6027646, 1.6034157
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3309300, 1.3301785
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1083813, 1.1087526
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4180393, 1.4188459

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6701041, upper bound: 0.6719641
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734529, upper bound: 0.6701031
time: 3.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 1, lower bound: -0.6719625, upper bound: 0.6665500
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 1, lower bound: -0.6650619, upper bound: 0.6734523
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 1, lower bound: -0.6701041, upper bound: 0.6719641
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.57
Output dim: 1, lower bound: -0.6734529, upper bound: 0.6701031

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7045913, 1.6981499
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3406433, 1.3436291
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2922716, 1.2935424
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9999912, 0.9948772
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3861008, 1.3940604
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1570902, 1.1577950
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6228092, 1.6248257
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3716116, 1.3673434
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1105211, 1.1103938
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4249523, 1.4234123

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6700990, upper bound: 0.6665470
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719597, upper bound: 0.6646572
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6984448, 1.7042959
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3432237, 1.3410490
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2927518, 1.2930622
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9952908, 0.9995775
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3935721, 1.3865893
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1580253, 1.1568596
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6254771, 1.6221578
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3665915, 1.3723636
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1107647, 1.1101501
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4242189, 1.4241455

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6645711, upper bound: 0.6734474
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6645713, upper bound: 0.6714727
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6531975, 1.6530671
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3210850, 1.3214259
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2879977, 1.2881742
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9652427, 0.9641275
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3579514, 1.3572516
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1636481, 1.1642927
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6017580, 1.6008325
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3296859, 1.3296945
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1072612, 1.1083235
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4174829, 1.4174223

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715765, upper bound: 0.6650601
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6646566, upper bound: 0.6719605
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6527717, 1.6534927
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3218317, 1.3206792
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2889647, 1.2872071
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9637138, 0.9656566
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3577402, 1.3574629
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1640623, 1.1638786
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6001811, 1.6024091
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3304465, 1.3289342
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1079521, 1.1076323
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4166155, 1.4182894

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6677805, upper bound: 0.6700935
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734433, upper bound: 0.6644311
time: 3.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6700990, upper bound: 0.6665470
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6719597, upper bound: 0.6646572
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6645711, upper bound: 0.6734474
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6645713, upper bound: 0.6714727
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6715765, upper bound: 0.6650601
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6646566, upper bound: 0.6719605
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6677805, upper bound: 0.6700935
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.57
Output dim: 1, lower bound: -0.6734433, upper bound: 0.6644311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7039061, 1.6978900
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3401695, 1.3424089
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2916648, 1.2919691
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9974895, 0.9939045
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3857515, 1.3939223
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1568143, 1.1571048
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6202254, 1.6238189
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3711271, 1.3660986
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1100919, 1.1092733
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4235287, 1.4228563

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6649064, upper bound: 0.6646514
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719522, upper bound: 0.6575123
time: 3.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6958141, 1.7006571
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3395742, 1.3384109
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2823913, 1.2855790
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9940038, 0.9977977
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3855126, 1.3807638
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1545513, 1.1520531
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6171188, 1.6105993
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3554926, 1.3643427
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1021932, 1.0982922
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4196930, 1.4178817

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6576170, upper bound: 0.6734416
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6645646, upper bound: 0.6663801
time: 3.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7040367, 1.6977599
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3398286, 1.3427498
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2914886, 1.2921455
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9986048, 0.9927890
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3864510, 1.3932228
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1561697, 1.1577494
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6211510, 1.6228933
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3711185, 1.3661070
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1090295, 1.1103355
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4235897, 1.4227953

Time for backsubstitution: 15.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715755, upper bound: 0.6650599
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6648867, upper bound: 0.6650592
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6978903, 1.7039061
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3424088, 1.3401697
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2919691, 1.2916651
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9939046, 0.9974893
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3939223, 1.3857515
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1571050, 1.1568143
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6238189, 1.6202254
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3660984, 1.3711269
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1092732, 1.1100919
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4228563, 1.4235287

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6627578, upper bound: 0.6714693
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6646522, upper bound: 0.6714688
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5721741, 1.5829635
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3203179, 1.3189490
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2454479, 1.2374644
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9460838, 0.9455154
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2967980, 1.2877970
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0791917, 1.0896208
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5835164, 1.5901834
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3605070, 1.3556292
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0865955, 1.0889381
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3658972, 1.3738930

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734423, upper bound: 0.6580184
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6670574, upper bound: 0.6644293
time: 3.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.20 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6649064, upper bound: 0.6646514
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6719522, upper bound: 0.6575123
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6576170, upper bound: 0.6734416
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6645646, upper bound: 0.6663801
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6715755, upper bound: 0.6650599
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6648867, upper bound: 0.6650592
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6627578, upper bound: 0.6714693
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6646522, upper bound: 0.6714688
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6734423, upper bound: 0.6580184
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.20
Output dim: 1, lower bound: -0.6670574, upper bound: 0.6644293

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6907387, 1.6954103
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3390723, 1.3365819
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2901914, 1.2841389
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9969463, 0.9909611
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3827534, 1.3780053
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1523080, 1.1562543
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6201339, 1.6233346
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3702741, 1.3612759
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1057827, 1.1084684
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4031448, 1.4190242

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719367, upper bound: 0.6570901
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6715254, upper bound: 0.6574966
time: 3.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6933336, 1.6874897
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3337475, 1.3373140
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2745616, 1.2841055
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9910603, 0.9972548
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3695951, 1.3777654
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1537008, 1.1475466
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6166344, 1.6105077
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3506694, 1.3634896
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1013882, 1.0939833
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4158611, 1.3974981

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6571963, upper bound: 0.6730124
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6571963, upper bound: 0.6734261
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7040877, 1.6978035
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3398492, 1.3427747
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2914944, 1.2921519
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9986168, 0.9927992
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3864760, 1.3932519
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1561868, 1.1577640
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6211953, 1.6229455
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3711410, 1.3661261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1090088, 1.1103117
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4236035, 1.4228075

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 849

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6644395, upper bound: 0.6650535
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715689, upper bound: 0.6579799
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5722256, 1.5830071
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3203387, 1.3189738
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2454531, 1.2374706
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9460708, 0.9455006
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2967634, 1.2877667
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0792086, 1.0896355
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5835602, 1.5902356
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3605292, 1.3556476
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0865748, 1.0889149
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3659067, 1.3739054

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6663734, upper bound: 0.6580119
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734353, upper bound: 0.6507955
time: 3.52 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.10 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6719367, upper bound: 0.6570901
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6715254, upper bound: 0.6574966
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6571963, upper bound: 0.6730124
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6571963, upper bound: 0.6734261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6644395, upper bound: 0.6650535
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6715689, upper bound: 0.6579799
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6663734, upper bound: 0.6580119
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 1, lower bound: -0.6734353, upper bound: 0.6507955

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6657934, 1.6669054
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3437495, 1.3407085
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2892437, 1.2833099
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9951634, 0.9889554
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3848157, 1.3803854
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1432598, 1.1483381
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5946856, 1.6010730
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3691959, 1.3603175
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1059264, 1.1086314
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4056129, 1.4218225

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719356, upper bound: 0.6570901
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6652654, upper bound: 0.6570898
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6683874, 1.6589854
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3384254, 1.3414407
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2736139, 1.2832763
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9892780, 0.9952495
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3716574, 1.3801458
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1446528, 1.1396315
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5911870, 1.5882466
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3495913, 1.3625309
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1015322, 1.0941465
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4183292, 1.4002962

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6556625, upper bound: 0.6730116
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6575993, upper bound: 0.6711405
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6648293, 1.6625450
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3378744, 1.3419912
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2737322, 1.2831576
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9890548, 0.9954721
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3719754, 1.3798277
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1457846, 1.1384987
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5943732, 1.5850604
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3497105, 1.3624113
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1015513, 1.0941272
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4186592, 1.3999665

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6571942, upper bound: 0.6667524
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6571945, upper bound: 0.6734230
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6909204, 1.6953235
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3387523, 1.3369480
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2900202, 1.2843213
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9980739, 0.9898559
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3834777, 1.3773346
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1516805, 1.1569138
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6211033, 1.6224610
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3702879, 1.3613026
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1046994, 1.1095066
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4032197, 1.4189756

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6659006, upper bound: 0.6579702
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715593, upper bound: 0.6523393
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5590572, 1.5805256
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3192418, 1.3131472
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2439780, 1.2296402
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9455179, 0.9425483
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2937739, 1.2718580
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0747027, 1.0887866
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5834687, 1.5897512
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3596666, 1.3508148
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0822644, 1.0881087
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3455229, 1.3700745

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734198, upper bound: 0.6503718
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730095, upper bound: 0.6507798
time: 3.48 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.64 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6719356, upper bound: 0.6570901
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6652654, upper bound: 0.6570898
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6556625, upper bound: 0.6730116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6575993, upper bound: 0.6711405
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6571942, upper bound: 0.6667524
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6571945, upper bound: 0.6734230
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6659006, upper bound: 0.6579702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6715593, upper bound: 0.6523393
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6734198, upper bound: 0.6503718
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 1, lower bound: -0.6730095, upper bound: 0.6507798

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6658454, 1.6669490
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3437705, 1.3407331
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2892492, 1.2833161
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9951756, 0.9889656
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3848403, 1.3804142
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1432772, 1.1483530
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5947304, 1.6011256
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3692164, 1.3603349
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1059060, 1.1086080
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4056273, 1.4218342

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6662631, upper bound: 0.6570819
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6719260, upper bound: 0.6514489
time: 3.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6681285, 1.6583004
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3372045, 1.3409667
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2720401, 1.2826698
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9883051, 0.9927474
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3715191, 1.3797963
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1439624, 1.1393552
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5901809, 1.5856636
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3483467, 1.3620470
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1004117, 1.0937172
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4177728, 1.3988726

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6556621, upper bound: 0.6663408
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6556625, upper bound: 0.6730110
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6648736, 1.6625969
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3378992, 1.3420123
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2737379, 1.2831624
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9890649, 0.9954839
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3720043, 1.3798525
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1457994, 1.1385159
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5944262, 1.5851048
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3497286, 1.3624322
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1015279, 1.0941066
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4186707, 1.3999805

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6552559, upper bound: 0.6734213
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6571926, upper bound: 0.6715500
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6103244, 1.6247954
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3372390, 1.3352185
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2465022, 1.2345786
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9804430, 0.9697139
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3225381, 1.3076711
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0668092, 1.0826565
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6044388, 1.6102362
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3879964
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0833421, 1.0908120
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3525009, 1.3745801

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6715437, upper bound: 0.6519148
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6711353, upper bound: 0.6523235
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5341289, 1.5520380
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3239192, 1.3172741
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2430308, 1.2288115
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9437356, 0.9405432
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2958333, 1.2742357
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0656474, 1.0808611
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5580072, 1.5674734
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3585873, 1.3498552
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0824084, 1.0882720
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3479910, 1.3728722

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734147, upper bound: 0.6499751
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6665135, upper bound: 0.6502039
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5305693, 1.5556045
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3233687, 1.3178251
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2431493, 1.2286930
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9435132, 0.9407660
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2961516, 1.2739177
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0667801, 1.0797310
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5611935, 1.5642900
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3587070, 1.3497355
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0824277, 1.0882528
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3483205, 1.3725424

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6710296, upper bound: 0.6503997
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730055, upper bound: 0.6503995
time: 3.56 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 21.80 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6662631, upper bound: 0.6570819
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6719260, upper bound: 0.6514489
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6556621, upper bound: 0.6663408
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6556625, upper bound: 0.6730110
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6552559, upper bound: 0.6734213
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6571926, upper bound: 0.6715500
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6715437, upper bound: 0.6519148
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6711353, upper bound: 0.6523235
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6734147, upper bound: 0.6499751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6665135, upper bound: 0.6502039
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6710296, upper bound: 0.6503997
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.80
Output dim: 1, lower bound: -0.6730055, upper bound: 0.6503995

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5852666, 1.5964386
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3422569, 1.3390034
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2457316, 1.2335739
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9775450, 0.9688240
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3238995, 1.3107500
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0583982, 1.0740865
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5780511, 1.5888829
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3870285
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0845481, 1.0899131
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3549080, 1.3774385

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6714341, upper bound: 0.6514449
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6714343, upper bound: 0.6496147
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6681719, 1.6583517
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3372295, 1.3409877
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2720461, 1.2826748
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9883153, 0.9927596
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3715484, 1.3798213
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1439772, 1.1393727
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5902328, 1.5857078
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3483644, 1.3620677
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1003885, 1.0936966
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4177847, 1.3988867

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6730000
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6556530, upper bound: 0.6673342
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6646128, 1.6619112
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3366785, 1.3415384
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2721643, 1.2825561
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9880922, 0.9929821
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3718665, 1.3795033
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1451092, 1.1382399
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5934196, 1.5825213
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3484840, 1.3619480
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1004076, 1.0936773
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4181147, 1.3985572

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6734119
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6552474, upper bound: 0.6677477
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6641874, 1.6623366
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3374255, 1.3407917
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2731314, 1.2815890
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9865634, 0.9945112
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3716552, 1.3797145
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1455233, 1.1378257
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5918427, 1.5840980
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3492441, 1.3611877
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1010985, 1.0929862
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4172473, 1.3994246

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6515500, upper bound: 0.6715404
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6571842, upper bound: 0.6658828
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5849710, 1.5967338
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3426627, 1.3385977
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2465222, 1.2327833
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9771314, 0.9692376
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3243878, 1.3102615
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0581677, 1.0743170
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5773997, 1.5895343
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3862770
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0841771, 1.0902841
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3541012, 1.3782451

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6714348, upper bound: 0.6496141
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734107, upper bound: 0.6496139
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5269303, 1.5529726
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3207306, 1.3141758
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2356658, 1.2183321
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9417336, 0.9394792
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.2903259, 1.2658582
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0619731, 1.0762568
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5496359, 1.5559325
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3506866, 1.3386364
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0705701, 1.0796813
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3420577, 1.3680164

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730004, upper bound: 0.6500226
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6661010, upper bound: 0.6502520
time: 3.67 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 21.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6714341, upper bound: 0.6514449
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6714343, upper bound: 0.6496147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6730000
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6556530, upper bound: 0.6673342
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6734119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6552474, upper bound: 0.6677477
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6515500, upper bound: 0.6715404
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6571842, upper bound: 0.6658828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6714348, upper bound: 0.6496141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6734107, upper bound: 0.6496139
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6730004, upper bound: 0.6500226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 21.93
Output dim: 1, lower bound: -0.6661010, upper bound: 0.6502520

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5976686, 1.5777724
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3354999, 1.3394743
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2223039, 1.2391570
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9681735, 0.9751292
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3018837, 1.3188803
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0697129, 1.0544939
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5779934, 1.5690286
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3750577, 1.3921270
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0816935, 1.0723389
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3733883, 1.3481669

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2454
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 1115
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2481
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 1261
type: RSZ, layer: 3, pos: 1110
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 3124
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 2567
type: RSZ, layer: 3, pos: 2474
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1487
type: RSZ, layer: 3, pos: 576
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 557
type: RSZ, layer: 3, pos: 1409
type: RSZ, layer: 3, pos: 1832
type: RSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2454

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6500209, upper bound: 0.6651597
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6422159, upper bound: 0.6730012
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5941019, 1.5813320
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3349487, 1.3400248
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2224224, 1.2390382
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9679506, 0.9753518
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3022017, 1.3185620
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0708430, 1.0533611
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5811768, 1.5658422
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3751774, 1.3920071
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0817126, 1.0723197
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3737183, 1.3478372

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2567
type: RSZ, layer: 3, pos: 2481
type: RSZ, layer: 3, pos: 1409
type: RSZ, layer: 3, pos: 1115
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 557
type: RSZ, layer: 3, pos: 1261
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 3124
type: RSZ, layer: 3, pos: 1487
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 2474
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 576
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1832
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 2454
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 1110
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6371122, upper bound: 0.6603019
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6371122, upper bound: 0.6604284
time: 3.56 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 22.43 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 22.43
Output dim: 1, lower bound: -0.6500209, upper bound: 0.6651597
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 22.43
Output dim: 1, lower bound: -0.6422159, upper bound: 0.6730012
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 22.43
Output dim: 1, lower bound: -0.6371122, upper bound: 0.6603019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 22.43
Output dim: 1, lower bound: -0.6371122, upper bound: 0.6604284
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 22.43
Output dim: 1, lower bound: -0.6734107, upper bound: 0.6496139
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 22.43
Output dim: 1, lower bound: -0.6730004, upper bound: 0.6500226
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2413.51 seconds
