## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.3120148312
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.5105915, -10.3324680, -12.5105915, -10.3324680, -2.1781235, 2.1781235)
1: (3.1649604, 4.4064875, 3.1649604, 4.4064875, -1.1255052, 1.1255053)
2: (-4.9406466, -3.7702384, -4.9406466, -3.7702384, -1.1704082, 1.1704082)
3: (-12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.7759883, 1.7759886)
4: (-2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.4544666, 1.4544665)
5: (-10.0812330, -8.6300726, -10.0812330, -8.6300726, -1.3241396, 1.3241397)
6: (-8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.7115004, 1.7115003)
7: (-2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.8193477, 0.8193477)
8: (-3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.4042225, 1.4042225)
9: (-12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.5232468, 1.5232465)

## BASE Result
execution time: IAR + LP analysis = 13.19 + 32.14 = 45.33 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.67 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.8663967847824097
rel_dist={1: [-0.5771277606023322, 0.5771254352696031]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.7368426322937012
rel_dist={1: [-0.38431871261449, 0.38431867244677376]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.6504731178283691
rel_dist={1: [-0.2375077651971229, 0.23750601401867089]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary Search Result
Binary search time: 188.79 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 3365.88 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=None

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509325, upper bound: 0.4509323
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509325, upper bound: 0.4509325
time: 3.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.72
Output dim: 1, lower bound: -0.4509325, upper bound: 0.4509323
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.72
Output dim: 1, lower bound: -0.4509325, upper bound: 0.4509325

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6372230, 1.6344876
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7801378, 0.7797550
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9188471, 0.9109509
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2819927, 1.2977731
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1220455, 1.1275221
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8818231, 0.8790795
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2499204, 1.2430710
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5886536, 0.5899296
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9850289, 0.9748379
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1032748, 1.1045026

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509261, upper bound: 0.4473148
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4473137, upper bound: 0.4509261
time: 3.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6344879, 1.6364419
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7797550, 0.7800273
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9109509, 0.9165723
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2932088, 1.2819927
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1259511, 1.1220454
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8790793, 0.8810382
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2430708, 1.2479552
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5895640, 0.5886536
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9748379, 0.9821028
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1041498, 1.1032749

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509228, upper bound: 0.4491965
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491966, upper bound: 0.4509226
time: 3.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.07
Output dim: 1, lower bound: -0.4509261, upper bound: 0.4473148
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.07
Output dim: 1, lower bound: -0.4473137, upper bound: 0.4509261
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.07
Output dim: 1, lower bound: -0.4509228, upper bound: 0.4491965
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.07
Output dim: 1, lower bound: -0.4491966, upper bound: 0.4509226

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6358516, 1.6351135
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7826315, 0.7743763
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9224417, 0.9032071
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2793186, 1.2990096
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1269681, 1.1168774
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8818576, 0.8790020
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2515094, 1.2396197
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5814903, 0.5932453
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9859802, 0.9727877
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0947981, 1.1084322

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509230, upper bound: 0.4426195
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4462260, upper bound: 0.4473106
time: 3.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6372230, 1.6331165
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7747591, 0.7797550
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9111032, 0.9109509
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2819927, 1.2950995
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1114008, 1.1275221
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8817451, 0.8790795
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2464690, 1.2430710
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5886536, 0.5827664
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9829789, 0.9748379
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1032748, 1.0960261

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432556, upper bound: 0.4509155
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4473020, upper bound: 0.4468668
time: 3.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6356850, 1.6381216
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7750304, 0.7743644
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8738649, 0.8856412
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2902260, 1.2776687
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1160038, 1.1137495
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8732063, 0.8728834
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2224649, 1.2227145
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5901883, 0.5917852
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9757438, 0.9807770
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1066122, 1.1051946

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509196, upper bound: 0.4444970
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4462231, upper bound: 0.4491933
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6361680, 1.6376381
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7740917, 0.7753029
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8800199, 0.8794862
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2888851, 1.2790089
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1176548, 1.1120985
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8709248, 0.8751649
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2178303, 1.2273494
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5926954, 0.5892781
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9735122, 0.9830084
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1060696, 1.1057372

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491902, upper bound: 0.4473041
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4455779, upper bound: 0.4509162
time: 3.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4509230, upper bound: 0.4426195
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4462260, upper bound: 0.4473106
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4432556, upper bound: 0.4509155
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4473020, upper bound: 0.4468668
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4509196, upper bound: 0.4444970
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4462231, upper bound: 0.4491933
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4491902, upper bound: 0.4473041
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.25
Output dim: 1, lower bound: -0.4455779, upper bound: 0.4509162

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6327527, 1.6286793
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7826314, 0.7744355
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9205763, 0.8993130
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2717788, 1.2953978
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1269658, 1.1181574
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8792431, 0.8777481
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2507544, 1.2380759
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5812159, 0.5922046
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9855601, 0.9719137
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0943696, 1.1082201

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509156, upper bound: 0.4404809
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4441030, upper bound: 0.4404817
time: 3.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6294177, 1.6320128
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7826908, 0.7743763
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9185476, 0.9013405
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2757075, 1.2914696
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1282470, 1.1168749
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8806040, 0.8763872
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2499654, 1.2388620
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5804497, 0.5929706
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9851059, 0.9723660
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0945859, 1.1080036

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440938, upper bound: 0.4404911
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440931, upper bound: 0.4473031
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6354229, 1.6319275
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7607530, 0.7704898
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9102120, 0.9096031
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2717643, 1.2796395
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1010926, 1.1119450
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8305695, 0.8452730
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2107356, 1.2194538
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5801868, 0.5699594
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9629045, 0.9444760
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0739617, 1.0766768

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456506
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
time: 3.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6360333, 1.6313171
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7654940, 0.7657489
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.9097555, 0.9100597
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2665324, 1.2848709
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0958230, 1.1172146
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8479388, 0.8279039
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2228518, 1.2073374
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5758466, 0.5742997
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9526169, 0.9547633
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0839257, 1.0667129

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472923, upper bound: 0.4451310
time: 3.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4455661, upper bound: 0.4468566
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6325839, 1.6316881
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7750301, 0.7744234
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8719983, 0.8817471
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2826848, 1.2740569
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1160014, 1.1150283
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8705920, 0.8716301
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2217073, 1.2211713
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5899139, 0.5907445
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9753217, 0.9799029
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1061835, 1.1049818

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468614, upper bound: 0.4444875
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509079, upper bound: 0.4404431
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6292503, 1.6350217
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7750895, 0.7743641
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8699708, 0.8837745
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2866135, 1.2701283
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1172826, 1.1137470
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8719529, 0.8702691
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2209210, 1.2219573
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5891476, 0.5915108
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9748694, 0.9803551
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1063995, 1.1047655

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4462163, upper bound: 0.4455748
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4426086, upper bound: 0.4491873
time: 3.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6347980, 1.6382651
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7765857, 0.7699243
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8836143, 0.8717420
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2862124, 1.2802463
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1225774, 1.1014535
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8709598, 0.8750875
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2194189, 1.2238983
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5855320, 0.5925934
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9744632, 0.9809582
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0975928, 1.1096666

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491871, upper bound: 0.4426085
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4444901, upper bound: 0.4473009
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6361680, 1.6362681
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7687132, 0.7753029
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8722758, 0.8794862
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2888851, 1.2763357
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1070101, 1.1120985
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8708472, 0.8751649
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2143787, 1.2273494
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5926954, 0.5821145
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9714620, 0.9830084
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1060696, 1.0972602

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4455743, upper bound: 0.4456514
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4403139, upper bound: 0.4509128
time: 3.41 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4509156, upper bound: 0.4404809
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4441030, upper bound: 0.4404817
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4440938, upper bound: 0.4404911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4440931, upper bound: 0.4473031
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456506
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4472923, upper bound: 0.4451310
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4455661, upper bound: 0.4468566
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4468614, upper bound: 0.4444875
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4509079, upper bound: 0.4404431
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4462163, upper bound: 0.4455748
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4426086, upper bound: 0.4491873
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4491871, upper bound: 0.4426085
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4444901, upper bound: 0.4473009
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4455743, upper bound: 0.4456514
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.33
Output dim: 1, lower bound: -0.4403139, upper bound: 0.4509128

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6191580, 1.6123686
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7782906, 0.7692318
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8960321, 0.8698678
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1895428, 1.2268357
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1258700, 1.1164705
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8650626, 0.8659306
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2486472, 1.2355480
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5818802, 0.5923290
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9861357, 0.9726719
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0936241, 1.1075990

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509058, upper bound: 0.4387448
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491797, upper bound: 0.4404734
time: 3.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6164415, 1.6150732
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7774278, 0.7699757
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8911312, 0.8747685
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2032204, 1.2131617
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1252789, 1.1158787
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8674248, 0.8635678
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2482266, 1.2359544
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5813402, 0.5928675
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9859533, 0.9724894
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0937414, 1.1074748

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440987, upper bound: 0.4352204
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4388392, upper bound: 0.4404757
time: 3.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6158121, 1.6157019
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7782310, 0.7691725
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8940036, 0.8718953
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1934719, 1.2229075
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1259682, 1.1151880
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8664237, 0.8645694
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2478464, 1.2363342
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5811130, 0.5930949
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9856818, 0.9727592
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0938406, 1.1073756

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440877, upper bound: 0.4352278
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4388328, upper bound: 0.4404865
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6131065, 1.6184182
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7774872, 0.7700353
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8891025, 0.8767976
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2071505, 1.2092335
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1265602, 1.1157793
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8687862, 0.8622069
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2474377, 1.2367549
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5805740, 0.5936352
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9858627, 0.9729416
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0939648, 1.1072581

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400340, upper bound: 0.4472936
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440814, upper bound: 0.4432471
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6179702, 1.6110005
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7509688, 0.7587489
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8854864, 0.8799365
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2812283, 1.2868721
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1067955, 1.1194115
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8382200, 0.8511190
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1943908, 1.1998550
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5594139, 0.5446237
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9761409, 0.9645427
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0615215, 1.0617566

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432423, upper bound: 0.4439159
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4415162, upper bound: 0.4456409
time: 3.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6144960, 1.6144748
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7490122, 0.7607057
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8805449, 0.8848758
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2789958, 1.2891045
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1085610, 1.1176469
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8364159, 0.8529229
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1911368, 1.2031077
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5548513, 0.5491856
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9829699, 0.9577127
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0590417, 1.0642364

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447306
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509037
time: 3.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6372304, 1.6329978
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7607696, 0.7600856
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8726695, 0.8791286
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2635484, 1.2805474
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0858757, 1.1089184
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8420658, 0.8197495
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2022451, 1.1820962
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5764711, 0.5774310
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9535223, 0.9534377
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0863881, 1.0686324

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472849, upper bound: 0.4389571
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4411189, upper bound: 0.4451238
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6377139, 1.6325145
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7598311, 0.7610243
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8788245, 0.8729736
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2622085, 1.2818875
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0875268, 1.1072674
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8397844, 0.8220310
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1976105, 1.1867309
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5789783, 0.5749239
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9512910, 0.9556690
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0858455, 1.0691750

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4455630, upper bound: 0.4421633
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4408707, upper bound: 0.4468543
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6307843, 1.6304986
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7610239, 0.7651582
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8711071, 0.8803993
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2724562, 1.2585969
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1056938, 1.0994509
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8194164, 0.8378236
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1859734, 1.1975534
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5814474, 0.5779376
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9552470, 0.9495405
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0768704, 1.0856327

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468542, upper bound: 0.4408720
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432427, upper bound: 0.4444786
time: 3.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6313946, 1.6298883
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7657650, 0.7604172
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8706508, 0.8808558
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2672243, 1.2638283
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1004241, 1.1047207
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8367854, 0.8204546
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1980896, 1.1854372
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5771070, 0.5822779
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9449598, 0.9598279
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0868344, 1.0756687

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509044, upper bound: 0.4351811
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456416, upper bound: 0.4404396
time: 3.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6278803, 1.6356483
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7775834, 0.7689854
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8735650, 0.8760304
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2839403, 1.2713649
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1222055, 1.1031023
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8719878, 0.8701917
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2225101, 1.2185062
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5819842, 0.5948261
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9758207, 0.9783050
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0979228, 1.1086950

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4421612, upper bound: 0.4455629
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4462046, upper bound: 0.4415166
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6292503, 1.6336513
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7697110, 0.7743641
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8622267, 0.8837745
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2866135, 1.2674546
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1066380, 1.1137470
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8718755, 0.8702691
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2174699, 1.2219573
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5891476, 0.5843472
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9728194, 0.9803551
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.1063995, 1.0962889

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4426032, upper bound: 0.4439226
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4373456, upper bound: 0.4491836
time: 3.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6316969, 1.6318314
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7765855, 0.7699834
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8817475, 0.8678479
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2786713, 1.2766337
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1225750, 1.1027325
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8683455, 0.8738341
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2186618, 1.2223548
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5852576, 0.5915527
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9740416, 0.9800841
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0971642, 1.1094539

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4451280, upper bound: 0.4425981
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491754, upper bound: 0.4385526
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6283634, 1.6351647
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7766447, 0.7699240
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8797200, 0.8698754
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2826004, 1.2727051
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1238563, 1.1014513
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8697064, 0.8724731
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2178755, 1.2231411
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5844914, 0.5923190
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9735891, 0.9805363
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0973802, 1.1092377

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4444847, upper bound: 0.4420351
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4356230, upper bound: 0.4472996
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6187148, 1.6153405
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7589288, 0.7635616
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8475482, 0.8498195
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2983487, 1.2835667
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1127129, 1.1195669
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8784974, 0.8810111
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1980314, 1.2077491
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5719220, 0.5567789
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9846982, 1.0030751
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0936289, 1.0823398

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4415162, upper bound: 0.4456409
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4455626, upper bound: 0.4415923
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6152406, 1.6188149
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7569720, 0.7655184
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8426087, 0.8547589
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2961161, 1.2857993
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1144781, 1.1178014
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8766935, 0.8828151
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1947789, 1.2110019
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5673595, 0.5613414
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9915284, 0.9962449
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0911489, 1.0848198

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4362557, upper bound: 0.4509022
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4403022, upper bound: 0.4468536
time: 4.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4509058, upper bound: 0.4387448
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4491797, upper bound: 0.4404734
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4440987, upper bound: 0.4352204
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4388392, upper bound: 0.4404757
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4440877, upper bound: 0.4352278
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4388328, upper bound: 0.4404865
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4400340, upper bound: 0.4472936
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4440814, upper bound: 0.4432471
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4432423, upper bound: 0.4439159
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4415162, upper bound: 0.4456409
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509037
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4472849, upper bound: 0.4389571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4411189, upper bound: 0.4451238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4455630, upper bound: 0.4421633
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4408707, upper bound: 0.4468543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4468542, upper bound: 0.4408720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4432427, upper bound: 0.4444786
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4509044, upper bound: 0.4351811
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4456416, upper bound: 0.4404396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4421612, upper bound: 0.4455629
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4462046, upper bound: 0.4415166
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4426032, upper bound: 0.4439226
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4373456, upper bound: 0.4491836
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4451280, upper bound: 0.4425981
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4491754, upper bound: 0.4385526
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4444847, upper bound: 0.4420351
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4356230, upper bound: 0.4472996
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4415162, upper bound: 0.4456409
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4455626, upper bound: 0.4415923
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4362557, upper bound: 0.4509022
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 1, lower bound: -0.4403022, upper bound: 0.4468536

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6203551, 1.6140490
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7735660, 0.7635686
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8589461, 0.8389368
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1865587, 1.2225120
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1159232, 1.1081746
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8591895, 0.8577759
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2280412, 1.2103075
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5825045, 0.5954604
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9870414, 0.9713459
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0960855, 1.1095179

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509024, upper bound: 0.4334834
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456398, upper bound: 0.4387395
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6208386, 1.6135657
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7726274, 0.7645073
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8651011, 0.8327818
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1852188, 1.2238522
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1175742, 1.1065235
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8569083, 0.8600574
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2234066, 1.2149423
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5850116, 0.5929533
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9848098, 0.9735774
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0955429, 1.1100606

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4451206, upper bound: 0.4404616
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4491680, upper bound: 0.4364152
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5989921, 1.5941455
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7676437, 0.7582351
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8664064, 0.8451024
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2126856, 1.2203946
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1309817, 1.1233478
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8750734, 0.8694128
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2318840, 1.2163560
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5605814, 0.5675454
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9991899, 0.9925561
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0812986, 1.0925536

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400396, upper bound: 0.4352108
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440870, upper bound: 0.4311621
time: 3.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5955145, 1.5976226
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7656871, 0.7601932
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8614650, 0.8500432
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2104535, 1.2226272
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1327469, 1.1215816
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8732696, 0.8712167
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2286282, 1.2196107
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5560182, 0.5721096
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0060155, 0.9857260
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0788202, 1.0950317

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4347800, upper bound: 0.4404640
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4388274, upper bound: 0.4364175
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5983617, 1.5947745
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7684484, 0.7574319
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8692780, 0.8422292
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2029366, 1.2301407
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1316712, 1.1226556
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8740730, 0.8704141
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2315040, 1.2167358
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5603549, 0.5677729
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9989183, 0.9928211
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0813975, 1.0924547

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440779, upper bound: 0.4334932
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4423518, upper bound: 0.4352181
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5948846, 1.5982523
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7664905, 0.7593887
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8643374, 0.8471699
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2007046, 1.2323730
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1334379, 1.1208909
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8722682, 0.8722180
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2282479, 1.2199913
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5557909, 0.5723362
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0057473, 0.9859958
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0789194, 1.0949330

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4388232, upper bound: 0.4387506
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4370981, upper bound: 0.4404768
time: 3.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6113069, 1.6172285
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7634810, 0.7607701
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8882113, 0.8754498
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1969223, 1.1937737
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1162519, 1.1002015
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8176109, 0.8284006
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2117040, 1.2131374
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5721073, 0.5808282
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9657881, 0.9425797
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0646517, 1.0879090

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400243, upper bound: 0.4455556
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4382981, upper bound: 0.4472839
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6119173, 1.6166182
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7682220, 0.7560291
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8877547, 0.8759062
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1916909, 1.1990051
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1109822, 1.1054710
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8349799, 0.8110315
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2238202, 1.2010212
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5677670, 0.5851685
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9555008, 0.9528670
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0746157, 1.0779450

Time for backsubstitution: 12.48 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.7800273895263672
rel_dist={1: [-0.4509328816559992, 0.4509326440462238]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3104051
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3128611
time: 4.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.30 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 8.30
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3104051
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.30
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3128611

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4602365, 1.4619739
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6819171, 0.6828954
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8175030, 0.8199725
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1797464, 1.1808627
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0504074, 1.0495245
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7761087, 0.7770107
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1124705, 1.1140970
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5067824, 0.5090636
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8894429, 0.8860277
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9844553, 0.9856954

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103367, upper bound: 0.3128606
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103367, upper bound: 0.3127808
time: 3.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.13 seconds
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.13
Output dim: 1, lower bound: -0.3103367, upper bound: 0.3128606
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.13
Output dim: 1, lower bound: -0.3103367, upper bound: 0.3127808

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4596508, 1.4600205
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6818360, 0.6826230
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8158298, 0.8143514
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1685288, 1.1775353
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0465019, 1.0483575
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7755220, 0.7750521
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1110111, 1.1092128
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5058720, 0.5087911
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8872728, 0.8787630
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9835806, 0.9854344

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3103286, upper bound: 0.3117975
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3092742, upper bound: 0.3128550
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4582832, 1.4613879
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6816446, 0.6828144
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8118818, 0.8183005
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1764195, 1.1696453
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0492399, 1.0456192
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7741501, 0.7764239
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1075863, 1.1126381
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5065101, 0.5081533
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8821778, 0.8838584
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9841943, 0.9848205

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3083868, upper bound: 0.3127744
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3103969, upper bound: 0.3107631
time: 3.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.32 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.32
Output dim: 1, lower bound: -0.3103286, upper bound: 0.3117975
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -0.3092742, upper bound: 0.3128550
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -0.3083868, upper bound: 0.3127744
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.32
Output dim: 1, lower bound: -0.3103969, upper bound: 0.3107631

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4610891, 1.4612172
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6761725, 0.6774287
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7818213, 0.7772654
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1642048, 1.1738811
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0373807, 1.0384109
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7673674, 0.7680385
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0857686, 1.0862877
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5077497, 0.5094151
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8859469, 0.8785524
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9854996, 0.9876246

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3067904, upper bound: 0.3103779
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3067904, upper bound: 0.3128538
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4564836, 1.4598932
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6676385, 0.6711787
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8107622, 0.8169527
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1635754, 1.1541858
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0362968, 1.0300415
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7229745, 0.7339329
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0718523, 1.0829623
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4958732, 0.4953464
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8569597, 0.8534964
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9548814, 0.9604895

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3072795, upper bound: 0.3117152
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3072795, upper bound: 0.3127721
time: 3.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.96 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 19.96
Output dim: 1, lower bound: -0.3067904, upper bound: 0.3103779
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.96
Output dim: 1, lower bound: -0.3067904, upper bound: 0.3128538
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 19.96
Output dim: 1, lower bound: -0.3072795, upper bound: 0.3117152
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.96
Output dim: 1, lower bound: -0.3072795, upper bound: 0.3127721

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4607172, 1.4598465
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6707939, 0.6759863
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7740772, 0.7751904
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1634867, 1.1712079
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0267360, 1.0355499
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7672899, 0.7680171
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0823174, 1.0853565
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5058259, 0.5022519
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8838965, 0.8780029
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9832261, 0.9791481

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3036277, upper bound: 0.3096875
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3036277, upper bound: 0.3128508
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4579225, 1.4610906
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6619747, 0.6659843
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7767539, 0.7798668
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1592515, 1.1505315
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0271759, 1.0200949
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7148204, 0.7269194
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0466100, 1.0600371
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4977510, 0.4959705
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8556340, 0.8532863
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9568005, 0.9626799

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3041048, upper bound: 0.3096142
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3041048, upper bound: 0.3127694
time: 3.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.93 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.93
Output dim: 1, lower bound: -0.3036277, upper bound: 0.3096875
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.93
Output dim: 1, lower bound: -0.3036277, upper bound: 0.3128508
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.93
Output dim: 1, lower bound: -0.3041048, upper bound: 0.3096142
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.93
Output dim: 1, lower bound: -0.3041048, upper bound: 0.3127694

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4444065, 1.4448898
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6655903, 0.6712147
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7446332, 0.7481976
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0880909, 1.0889735
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0250490, 1.0341585
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7542892, 0.7538354
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0797894, 1.0830327
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5049271, 0.5016234
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8845640, 0.8785788
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9825387, 0.9784021

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3016476, upper bound: 0.3128457
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3036194, upper bound: 0.3108341
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4416108, 1.4461331
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6567709, 0.6612126
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7473094, 0.7528727
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0838537, 1.0682974
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0254886, 1.0187032
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7018199, 0.7127376
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0440819, 1.0577147
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4968523, 0.4953420
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8563006, 0.8538618
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9561133, 0.9619342

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3037408, upper bound: 0.3091725
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3037379, upper bound: 0.3127609
time: 3.75 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.90 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.90
Output dim: 1, lower bound: -0.3016476, upper bound: 0.3128457
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 19.90
Output dim: 1, lower bound: -0.3036194, upper bound: 0.3108341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 19.90
Output dim: 1, lower bound: -0.3037408, upper bound: 0.3091725
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.90
Output dim: 1, lower bound: -0.3037379, upper bound: 0.3127609

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4426064, 1.4433947
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6515839, 0.6595789
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7435133, 0.7468497
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0752473, 1.0735137
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0121064, 1.0185812
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7031138, 0.7113445
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0440555, 1.0533569
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4942905, 0.4888167
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8593457, 0.8482170
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9532259, 0.9540714

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3012076, upper bound: 0.3092389
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3012043, upper bound: 0.3128470
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4351766, 1.4413724
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6568005, 0.6612124
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7434151, 0.7499927
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0782771, 1.0607564
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0261283, 1.0187007
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.6998856, 0.7101228
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0425382, 1.0565712
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4968483, 0.4957216
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8554260, 0.8532140
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9557965, 0.9615057

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3102862
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3127562
time: 3.57 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 19.96 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 19.96
Output dim: 1, lower bound: -0.3012076, upper bound: 0.3092389
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.96
Output dim: 1, lower bound: -0.3012043, upper bound: 0.3128470
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 19.96
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3102862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.96
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3127562

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4361718, 1.4386332
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6516135, 0.6595788
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7396193, 0.7439700
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0696721, 1.0659735
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0127454, 1.0185786
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7011795, 0.7087297
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0425119, 1.0522139
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4942865, 0.4891963
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8584710, 0.8475684
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9529090, 0.9536427

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1500
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 2487
type: RSZ, layer: 3, pos: 899
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2476
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2144
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1488
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 1188
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 1151
type: RSZ, layer: 3, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1500

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075199
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075440
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4348042, 1.4400017
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6514220, 0.6597703
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7356713, 0.7479180
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0775590, 1.0580833
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0154841, 1.0158404
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.6998081, 0.7101015
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0390871, 1.0556402
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4949245, 0.4885583
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8533760, 0.8526648
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9535229, 0.9530288

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1500
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 2476
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1188
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 2487
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 2144
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1488
type: RSZ, layer: 3, pos: 899
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1151
type: RSZ, layer: 3, pos: 668

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1500

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075199
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2962288, upper bound: 0.3075445
time: 3.84 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 20.29 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 20.29
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075199
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 20.29
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 20.29
Output dim: 1, lower bound: -0.2962275, upper bound: 0.3075199
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 20.29
Output dim: 1, lower bound: -0.2962288, upper bound: 0.3075445
Binary search (step 2): status=Status.VERIFIED, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary search (step 3) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843100, upper bound: 0.3808805
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808811, upper bound: 0.3843094
time: 3.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.74
Output dim: 1, lower bound: -0.3843100, upper bound: 0.3808805
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.74
Output dim: 1, lower bound: -0.3808811, upper bound: 0.3843094

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5574322, 1.5589299
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7373682, 0.7314639
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8826308, 0.8741269
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2301891, 1.2331216
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0859172, 1.0742418
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8256571, 0.8255728
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1903410, 1.1865610
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5536778, 0.5615370
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9276477, 0.9253968
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0432863, 1.0525911

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843072, upper bound: 0.3772554
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3806341, upper bound: 0.3808782
time: 3.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5588031, 1.5574322
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7314639, 0.7368426
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8741269, 0.8818710
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2328613, 1.2301888
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0742416, 1.0848864
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8255730, 0.8256506
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1865609, 1.1900121
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5608410, 0.5536778
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9253968, 0.9274467
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0517628, 1.0432863

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808784, upper bound: 0.3806335
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772556, upper bound: 0.3843063
time: 3.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.08
Output dim: 1, lower bound: -0.3843072, upper bound: 0.3772554
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.08
Output dim: 1, lower bound: -0.3806341, upper bound: 0.3808782
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.08
Output dim: 1, lower bound: -0.3808784, upper bound: 0.3806335
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.08
Output dim: 1, lower bound: -0.3772556, upper bound: 0.3843063

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5534978, 1.5524952
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7373682, 0.7315084
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8802574, 0.8702328
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2226479, 1.2285271
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0859149, 1.0752003
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8230426, 0.8239790
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1893872, 1.1850172
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5532118, 0.5604963
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9271127, 0.9245225
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0428574, 1.0523243

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843046, upper bound: 0.3734067
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804723, upper bound: 0.3772516
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5509973, 1.5549953
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7374128, 0.7314639
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8787367, 0.8717535
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2255943, 1.2255807
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0868759, 1.0742393
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8240633, 0.8229582
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1887974, 1.1856071
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5526371, 0.5610709
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9267734, 0.9248618
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0430195, 1.0521619

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3806299, upper bound: 0.3770258
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3734067, upper bound: 0.3808777
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5548687, 1.5509975
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7314639, 0.7368871
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8717535, 0.8779769
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2253215, 1.2255945
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0742393, 1.0858450
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8229582, 0.8240566
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1856068, 1.1884686
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5603750, 0.5526371
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9248618, 0.9265726
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0513341, 1.0430195

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808706, upper bound: 0.3792848
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3795365, upper bound: 0.3806259
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5523686, 1.5534976
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7315084, 0.7368425
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8702328, 0.8794975
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2282679, 1.2226479
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0752003, 1.0848842
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8239789, 0.8230360
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1850172, 1.1890582
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5598003, 0.5532118
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9245225, 0.9269117
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0514963, 1.0428574

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772479, upper bound: 0.3829575
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3759132, upper bound: 0.3842992
time: 3.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3843046, upper bound: 0.3734067
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3804723, upper bound: 0.3772516
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3806299, upper bound: 0.3770258
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3734067, upper bound: 0.3808777
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3808706, upper bound: 0.3792848
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3795365, upper bound: 0.3806259
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3772479, upper bound: 0.3829575
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.07
Output dim: 1, lower bound: -0.3759132, upper bound: 0.3842992

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5351791, 1.5315685
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7270948, 0.7197673
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8542957, 0.8405659
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2315538, 1.2357585
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0916176, 1.0822279
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8302410, 0.8298243
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1722302, 1.1654190
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5313030, 0.5351652
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9403495, 0.9428819
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0297971, 1.0374041

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843044, upper bound: 0.3734030
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843044, upper bound: 0.3734030
time: 3.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5325708, 1.5341744
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7256272, 0.7212350
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8505902, 0.8442702
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2298791, 1.2374327
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0929418, 1.0809031
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8288877, 0.8311774
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1697886, 1.1678585
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5278807, 0.5385871
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9454683, 0.9377593
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0279369, 1.0392625

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804721, upper bound: 0.3772505
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804721, upper bound: 0.3772505
time: 3.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5326767, 1.5340686
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7271392, 0.7197229
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8527741, 0.8420864
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2345002, 1.2328119
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0925786, 1.0812659
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8312614, 0.8288037
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1716383, 1.1660086
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5307279, 0.5357399
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9400102, 0.9432174
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0299578, 1.0372417

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790115, upper bound: 0.3717854
time: 3.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790093, upper bound: 0.3770200
time: 3.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5300708, 1.5366769
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7256718, 0.7211905
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8490696, 0.8457921
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2328260, 1.2344863
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0939035, 1.0799420
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8299086, 0.8301566
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1691988, 1.1684502
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5273060, 0.5391623
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9451329, 0.9380984
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0280993, 1.0391016

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3752059, upper bound: 0.3755982
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3752012, upper bound: 0.3808706
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5560660, 1.5525582
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7265046, 0.7312238
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8346672, 0.8455071
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2220020, 1.2212706
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0642922, 1.0771360
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8165150, 0.8159021
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1638426, 1.1632282
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5609993, 0.5551416
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9252096, 0.9252470
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0536604, 1.0449392

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777036, upper bound: 0.3792759
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808614, upper bound: 0.3760598
time: 3.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5564284, 1.5521958
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7258006, 0.7319278
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8392837, 0.8408909
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2209969, 1.2222757
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0655305, 1.0758979
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8148041, 0.8176132
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1603664, 1.1667041
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5628797, 0.5532613
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9235359, 0.9269205
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0532537, 1.0453461

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3795363, upper bound: 0.3806253
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3795363, upper bound: 0.3806253
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5535660, 1.5550585
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7265491, 0.7311794
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8331468, 0.8470278
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2249489, 1.2183239
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0652533, 1.0761752
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8175359, 0.8148814
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1632529, 1.1638178
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5604247, 0.5557163
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9248704, 0.9255861
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0538225, 1.0447768

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3740591, upper bound: 0.3829485
time: 3.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772386, upper bound: 0.3797322
time: 3.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5539284, 1.5546958
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7258452, 0.7318833
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8377631, 0.8424115
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2239437, 1.2193291
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0664914, 1.0749369
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8158245, 0.8165926
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1597768, 1.1672937
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5623050, 0.5538359
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9231967, 0.9272598
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0534158, 1.0451838

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3759068, upper bound: 0.3804641
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3720611, upper bound: 0.3842959
time: 3.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3843044, upper bound: 0.3734030
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3843044, upper bound: 0.3734030
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3804721, upper bound: 0.3772505
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3804721, upper bound: 0.3772505
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3790115, upper bound: 0.3717854
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3790093, upper bound: 0.3770200
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3752059, upper bound: 0.3755982
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3752012, upper bound: 0.3808706
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3777036, upper bound: 0.3792759
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3808614, upper bound: 0.3760598
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3795363, upper bound: 0.3806253
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3795363, upper bound: 0.3806253
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3740591, upper bound: 0.3829485
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3772386, upper bound: 0.3797322
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3759068, upper bound: 0.3804641
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.17
Output dim: 1, lower bound: -0.3720611, upper bound: 0.3842959

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5352788, 1.5296152
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7271094, 0.7194951
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8545983, 0.8349447
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2203381, 1.2363775
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0877123, 1.0824307
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8303400, 0.8278658
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1724854, 1.1605345
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5303928, 0.5352119
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9407290, 0.9356169
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0289221, 1.0374500

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842999, upper bound: 0.3717799
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790142, upper bound: 0.3717829
time: 3.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5332260, 1.5315685
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7268224, 0.7197673
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8486748, 0.8405659
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2315538, 1.2245426
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0916176, 1.0783228
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8282825, 0.8298243
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1673458, 1.1654190
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5313030, 0.5342549
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9330844, 0.9428819
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0297971, 1.0365289

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842999, upper bound: 0.3717798
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790142, upper bound: 0.3717829
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5326705, 1.5322211
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7256418, 0.7209626
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8508923, 0.8386492
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2186635, 1.2380519
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0890365, 1.0811061
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8289870, 0.8292189
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1700432, 1.1629740
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5269704, 0.5386335
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9458481, 0.9304942
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0270619, 1.0393084

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804644, upper bound: 0.3759058
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3791236, upper bound: 0.3772428
time: 3.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5306177, 1.5341744
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7253548, 0.7212350
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8449693, 0.8442702
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2298791, 1.2262168
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0929418, 1.0769976
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8269295, 0.8311774
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1649041, 1.1678585
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5278807, 0.5376768
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9382032, 0.9377593
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0279369, 1.0383874

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804676, upper bound: 0.3755905
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3752048, upper bound: 0.3755914
time: 3.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5183954, 1.5177572
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7224948, 0.7145196
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8270068, 0.8126425
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1522655, 1.1608362
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0904481, 1.0795794
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8170807, 0.8163946
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1694171, 1.1634808
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5312663, 0.5358729
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9405859, 0.9436564
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0292120, 1.0365835

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790102, upper bound: 0.3717851
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790102, upper bound: 0.3717852
time: 3.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5163651, 1.5197942
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7219360, 0.7151666
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8233299, 0.8163191
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1625252, 1.1505775
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0908918, 1.0800228
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8188527, 0.8146226
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1691105, 1.1637964
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5308610, 0.5362779
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9407227, 0.9437932
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0293050, 1.0364958

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790085, upper bound: 0.3770225
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790085, upper bound: 0.3770225
time: 3.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5157876, 1.5203655
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7210263, 0.7159872
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8233013, 0.8163480
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1505914, 1.1625102
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0917730, 1.0782554
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8157277, 0.8177474
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1669760, 1.1659224
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5278434, 0.5392954
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9457085, 0.9385375
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0273533, 1.0384423

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3751985, upper bound: 0.3742542
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3738570, upper bound: 0.3755898
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5137591, 1.5224028
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7204684, 0.7166352
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8196259, 0.8200248
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1608505, 1.1522522
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0922170, 1.0786989
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8174996, 0.8159757
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1666710, 1.1662378
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5274391, 0.5397007
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9458454, 0.9386742
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0274463, 1.0383557

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3719757, upper bound: 0.3808636
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3751920, upper bound: 0.3777044
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5542665, 1.5512159
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7124984, 0.7207734
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8336620, 0.8441594
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2104654, 1.2058103
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0526664, 1.0615587
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7653399, 0.7777535
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1281084, 1.1365812
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5514477, 0.5423348
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9025631, 0.8948847
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0243475, 1.0230992

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777034, upper bound: 0.3792751
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777034, upper bound: 0.3792751
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5547242, 1.5507581
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7160542, 0.7172176
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8333199, 0.8445017
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2065420, 1.2097337
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0487142, 1.0655110
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7783666, 0.7647266
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1371957, 1.1274941
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5481926, 0.5455899
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8948474, 0.9026002
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0318205, 1.0156262

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808612, upper bound: 0.3760588
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808612, upper bound: 0.3760588
time: 3.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5565269, 1.5502415
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7258153, 0.7316552
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8395853, 0.8352695
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2097807, 1.2228937
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0616252, 1.0761009
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8149030, 0.8156548
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1606207, 1.1618193
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5619695, 0.5533079
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9239160, 0.9196557
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0523782, 1.0453920

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3763659, upper bound: 0.3806162
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3795270, upper bound: 0.3774003
time: 3.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5544746, 1.5521958
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7255282, 0.7319278
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8336623, 0.8408909
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2209969, 1.2110589
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0655305, 1.0719925
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8128455, 0.8176132
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1554818, 1.1667041
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5628797, 0.5523508
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9162713, 0.9269205
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0532537, 1.0444710

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3795313, upper bound: 0.3790046
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3742572, upper bound: 0.3790070
time: 3.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5517659, 1.5537162
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7125430, 0.7207288
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8321416, 0.8456800
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2134123, 1.2028637
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0536275, 1.0605977
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7663603, 0.7767327
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1275188, 1.1371709
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5508731, 0.5429094
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9022238, 0.8952239
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0245097, 1.0229371

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3724187, upper bound: 0.3776597
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3724158, upper bound: 0.3829459
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5522237, 1.5532584
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7160987, 0.7171732
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8317993, 0.8460222
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2094884, 1.2067873
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0496752, 1.0645502
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7793870, 0.7637060
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1366061, 1.1280837
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5476179, 0.5461646
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8945084, 0.9029393
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0319827, 1.0154638

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772345, upper bound: 0.3758977
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3733873, upper bound: 0.3797294
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5356069, 1.5337679
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7155713, 0.7201418
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8118005, 0.8127446
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2328489, 1.2265606
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0721941, 1.0819635
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8230231, 0.8224380
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1426158, 1.1476934
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5403956, 0.5285048
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9364326, 0.9456148
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0403540, 1.0302634

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3727171, upper bound: 0.3804546
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3758976, upper bound: 0.3772389
time: 3.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5330009, 1.5363762
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7141037, 0.7216094
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8080959, 0.8164501
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2311747, 1.2282350
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0735192, 1.0806396
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8216701, 0.8237910
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1401763, 1.1501348
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5369737, 0.5319272
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9415553, 0.9404957
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0384953, 1.0321236

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3704412, upper bound: 0.3790065
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3704377, upper bound: 0.3842947
time: 3.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3842999, upper bound: 0.3717799
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790142, upper bound: 0.3717829
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3842999, upper bound: 0.3717798
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790142, upper bound: 0.3717829
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3804644, upper bound: 0.3759058
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3791236, upper bound: 0.3772428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3804676, upper bound: 0.3755905
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3752048, upper bound: 0.3755914
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790102, upper bound: 0.3717851
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790102, upper bound: 0.3717852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790085, upper bound: 0.3770225
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3790085, upper bound: 0.3770225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3751985, upper bound: 0.3742542
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3738570, upper bound: 0.3755898
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3719757, upper bound: 0.3808636
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3751920, upper bound: 0.3777044
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3777034, upper bound: 0.3792751
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3777034, upper bound: 0.3792751
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3808612, upper bound: 0.3760588
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3808612, upper bound: 0.3760588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3763659, upper bound: 0.3806162
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3795270, upper bound: 0.3774003
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3795313, upper bound: 0.3790046
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3742572, upper bound: 0.3790070
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3724187, upper bound: 0.3776597
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3724158, upper bound: 0.3829459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3772345, upper bound: 0.3758977
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3733873, upper bound: 0.3797294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3727171, upper bound: 0.3804546
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3758976, upper bound: 0.3772389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3704412, upper bound: 0.3790065
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.10
Output dim: 1, lower bound: -0.3704377, upper bound: 0.3842947

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5210040, 1.5133030
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7225542, 0.7142916
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8288293, 0.8055005
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1381025, 1.1643982
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0864689, 1.0807438
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8161592, 0.8154567
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1702728, 1.1580064
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5309315, 0.5353452
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9413049, 0.9363296
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0281761, 1.0367970

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842922, upper bound: 0.3704365
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3829511, upper bound: 0.3717722
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5189664, 1.5153315
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7219061, 0.7148495
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8251538, 0.8091761
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1483607, 1.1541426
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0860255, 1.0803001
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8179307, 0.8136848
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1699572, 1.1583111
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5305260, 0.5357492
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9411681, 0.9361926
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0282626, 1.0367041

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790065, upper bound: 0.3704395
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3776654, upper bound: 0.3717755
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5189512, 1.5152571
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7222672, 0.7145641
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8229074, 0.8111218
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1493196, 1.1525669
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0903742, 1.0766358
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8141012, 0.8174152
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1651335, 1.1628911
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5318415, 0.5343882
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9336603, 0.9435946
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0290508, 1.0358763

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842922, upper bound: 0.3704365
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3829511, upper bound: 0.3717722
time: 3.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5169137, 1.5172856
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7216190, 0.7151220
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8192306, 0.8147974
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1595774, 1.1423075
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0899310, 1.0761919
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8158731, 0.8156434
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1648178, 1.1631961
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5314362, 0.5347924
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9335234, 0.9434577
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0291376, 1.0357831

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757888, upper bound: 0.3717739
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790050, upper bound: 0.3686130
time: 3.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5338674, 1.5337803
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7206821, 0.7152989
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8138061, 0.8061793
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2153442, 1.2337277
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0790894, 1.0723975
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8225436, 0.8210642
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1482767, 1.1377313
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5275945, 0.5411379
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9461956, 0.9291682
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0293882, 1.0412276

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772393, upper bound: 0.3758989
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804552, upper bound: 0.3727164
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5342298, 1.5334179
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7199782, 0.7160029
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8184223, 0.8015630
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2143390, 1.2347329
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0803275, 1.0711591
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8208325, 0.8227754
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1448005, 1.1412072
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5294749, 0.5392575
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9445221, 0.9308418
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0289812, 1.0416346

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3758983, upper bound: 0.3772358
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3791144, upper bound: 0.3740542
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5163429, 1.5178628
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7207985, 0.7160317
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8192015, 0.8148264
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1476450, 1.1542411
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0916984, 1.0753107
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8127484, 0.8187683
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1626916, 1.1653306
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5284187, 0.5378101
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9387794, 0.9384719
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0271912, 1.0377347

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772426, upper bound: 0.3755811
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804584, upper bound: 0.3724187
time: 3.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5143054, 1.5198934
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7201514, 0.7165905
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8155251, 0.8185030
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1579032, 1.1439819
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0912549, 1.0748668
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8145204, 0.8169963
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1623762, 1.1656370
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5280137, 0.5382154
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9386425, 0.9383351
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0272789, 1.0376415

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3719794, upper bound: 0.3755824
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3751956, upper bound: 0.3724205
time: 3.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5184934, 1.5158031
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7225095, 0.7142471
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8273077, 0.8070211
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1410489, 1.1614521
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0865424, 1.0797808
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8171802, 0.8144358
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1696723, 1.1585960
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5303563, 0.5359197
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9409642, 0.9363915
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0283368, 1.0366297

Time for backsubstitution: 12.74 seconds
Binary search (step 3): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.7368426322937012
rel_dist={1: [-0.38431871261449, 0.38431867244677376]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1965.17 seconds
