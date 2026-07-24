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
execution time: IAR + LP analysis = 13.17 + 31.79 = 44.96 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3555.04 seconds, max iter: 100)

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
Binary search time: 187.56 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_dual_Z) starts
Time budget: 3367.49 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=None

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509293, upper bound: 0.4456677
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456678, upper bound: 0.4509292
time: 3.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.75
Output dim: 1, lower bound: -0.4509293, upper bound: 0.4456677
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.75
Output dim: 1, lower bound: -0.4456678, upper bound: 0.4509292

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6189885, 1.6155138
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7702433, 0.7682865
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8918445, 0.8869053
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3026741, 1.3004413
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1316535, 1.1334188
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8886881, 0.8868842
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2316095, 1.2283570
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5687909, 0.5642283
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9953396, 1.0021696
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0917096, 1.0892296

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509230, upper bound: 0.4420502
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4473106, upper bound: 0.4456617
time: 3.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6155138, 1.6189883
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7682866, 0.7702434
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8869052, 0.8918445
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3004415, 1.3026738
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1334188, 1.1316535
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8868840, 0.8886881
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2283567, 1.2316097
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5642283, 0.5687908
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0021696, 0.9953395
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0892296, 1.0917096

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456616, upper bound: 0.4473116
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420490, upper bound: 0.4509229
time: 3.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.12
Output dim: 1, lower bound: -0.4509230, upper bound: 0.4420502
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.12
Output dim: 1, lower bound: -0.4473106, upper bound: 0.4456617
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.12
Output dim: 1, lower bound: -0.4456616, upper bound: 0.4473116
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.12
Output dim: 1, lower bound: -0.4420490, upper bound: 0.4509229

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6176176, 1.6161404
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7727371, 0.7629080
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8954391, 0.8791615
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3000004, 1.3016782
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1365764, 1.1227744
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8887231, 0.8868066
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2331989, 1.2249061
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5616275, 0.5675439
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9962903, 1.0001193
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0832329, 1.0931591

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509226, upper bound: 0.4420497
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509226, upper bound: 0.4420497
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6189885, 1.6141434
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7648647, 0.7682865
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8841007, 0.8869053
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3026741, 1.2977679
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1210089, 1.1334188
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8886106, 0.8868842
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2281587, 1.2283570
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5687909, 0.5570650
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9932891, 1.0021696
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0917096, 1.0807530

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4473102, upper bound: 0.4456613
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4473102, upper bound: 0.4456613
time: 3.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6141434, 1.6196146
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7707803, 0.7648647
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8904998, 0.8841007
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2977679, 1.3039107
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1383417, 1.1210089
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8869188, 0.8886106
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2299464, 1.2281588
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5570650, 0.5721065
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0031205, 0.9932891
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0807528, 1.0956391

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456612, upper bound: 0.4473113
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456612, upper bound: 0.4473100
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6155138, 1.6176176
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7629080, 0.7702434
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8791614, 0.8918445
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3004415, 1.3000004
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1227741, 1.1316535
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8868067, 0.8886881
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2249062, 1.2316097
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5642283, 0.5616275
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0001193, 0.9953395
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0892296, 1.0832328

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420485, upper bound: 0.4509237
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420485, upper bound: 0.4509226
time: 3.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4509226, upper bound: 0.4420497
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4509226, upper bound: 0.4420497
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4473102, upper bound: 0.4456613
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4473102, upper bound: 0.4456613
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4456612, upper bound: 0.4473113
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4456612, upper bound: 0.4473100
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4420485, upper bound: 0.4509237
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.36
Output dim: 1, lower bound: -0.4420485, upper bound: 0.4509226

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6183994, 1.6141868
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7728474, 0.7626355
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8977160, 0.8735402
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2887838, 1.3062420
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1326709, 1.1243448
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8895081, 0.8848480
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2351646, 1.2200212
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5607172, 0.5679096
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9992169, 0.9928550
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0823579, 1.0935118

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468635, upper bound: 0.4420380
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509109, upper bound: 0.4379915
time: 3.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6156642, 1.6161404
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7724648, 0.7629080
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8898180, 0.8791615
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3000004, 1.2904615
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1365764, 1.1188688
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8867644, 0.8868066
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2283142, 1.2249061
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5616275, 0.5666336
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9890260, 1.0001193
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0832329, 1.0922842

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468635, upper bound: 0.4420380
time: 3.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509109, upper bound: 0.4379915
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6197698, 1.6121898
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7649750, 0.7680141
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8863776, 0.8812842
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2914565, 1.3023317
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1171036, 1.1349895
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8893956, 0.8849255
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2301245, 1.2234727
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5678805, 0.5574307
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9962157, 0.9949049
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0908346, 1.0811057

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456506
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472985, upper bound: 0.4416032
time: 3.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6170352, 1.6141434
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7645923, 0.7682865
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8784792, 0.8869053
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3026741, 1.2865512
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1210089, 1.1295135
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8866521, 0.8868842
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2232740, 1.2283570
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5687909, 0.5561546
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9860247, 1.0021696
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0917096, 1.0798779

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456494
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472985, upper bound: 0.4416022
time: 3.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6149251, 1.6176612
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7708908, 0.7645923
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8927748, 0.8784794
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2865512, 1.3084741
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1344364, 1.1225803
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8877038, 0.8866520
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2319109, 1.2232739
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5561547, 0.5724717
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0060459, 0.9860247
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0798779, 1.0959918

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4416021, upper bound: 0.4472996
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456495, upper bound: 0.4432531
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6121900, 1.6196146
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7705079, 0.7648647
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8848784, 0.8841007
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2977679, 1.2926941
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1383417, 1.1171036
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8849605, 0.8886106
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2250614, 1.2281588
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5570650, 0.5711962
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9958562, 0.9932891
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0807528, 1.0947640

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4416021, upper bound: 0.4472985
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456495, upper bound: 0.4432531
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6162956, 1.6156642
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7630184, 0.7699709
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8814363, 0.8862236
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2892239, 1.3045640
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1188688, 1.1332247
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8875915, 0.8867295
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2268708, 1.2267255
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5633180, 0.5619928
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.0030447, 0.9880747
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0883546, 1.0835855

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420368, upper bound: 0.4468635
time: 3.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6135604, 1.6176176
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7626356, 0.7702434
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8735402, 0.8918445
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.3004415, 1.2887838
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1227741, 1.1277480
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8848480, 0.8886881
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2200212, 1.2316097
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5642283, 0.5607172
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9928550, 0.9953395
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0892296, 1.0823579

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420368, upper bound: 0.4468635
time: 3.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4468635, upper bound: 0.4420380
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4509109, upper bound: 0.4379915
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4468635, upper bound: 0.4420380
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4509109, upper bound: 0.4379915
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456506
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4472985, upper bound: 0.4416032
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4432521, upper bound: 0.4456494
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4472985, upper bound: 0.4416022
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4416021, upper bound: 0.4472996
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4456495, upper bound: 0.4432531
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4416021, upper bound: 0.4472985
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4456495, upper bound: 0.4432531
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4420368, upper bound: 0.4468635
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4379904, upper bound: 0.4509108
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.51
Output dim: 1, lower bound: -0.4420368, upper bound: 0.4468635

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6165993, 1.6129973
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7588413, 0.7533704
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8968246, 0.8721924
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2785556, 1.2907825
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1223631, 1.1087673
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8383323, 0.8510416
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1994309, 1.1964036
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5522505, 0.5551027
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9791421, 0.9624926
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0530448, 1.0741627

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468561, upper bound: 0.4358655
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4406880, upper bound: 0.4420292
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6172097, 1.6123872
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7635823, 0.7486293
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8963683, 0.8726487
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2733243, 1.2960138
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1170933, 1.1140370
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8557016, 0.8336725
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2115471, 1.1842874
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5479102, 0.5594430
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9688548, 0.9727800
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0630088, 1.0641987

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509035, upper bound: 0.4318201
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4447305, upper bound: 0.4379826
time: 3.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6138647, 1.6149507
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7584586, 0.7536428
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8889265, 0.8778135
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2897723, 1.2750020
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1262684, 1.1032913
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8355888, 0.8530002
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1925802, 1.2012886
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5531609, 0.5538267
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9689511, 0.9697577
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0539198, 1.0729351

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468561, upper bound: 0.4358655
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4406880, upper bound: 0.4420292
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6144750, 1.6143403
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7631997, 0.7489018
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8884702, 0.8782699
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2845409, 1.2802334
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1209989, 1.1085608
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8529578, 0.8356310
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2046967, 1.1891723
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5488205, 0.5581670
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9586638, 0.9800450
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0638838, 1.0629711

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509035, upper bound: 0.4318201
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4447305, upper bound: 0.4379826
time: 3.47 seconds

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

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432446, upper bound: 0.4394739
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4370773, upper bound: 0.4456419
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6185806, 1.6103902
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7557099, 0.7540079
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8850298, 0.8803929
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2759974, 1.2921035
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1015258, 1.1246812
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8555890, 0.8337499
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2065070, 1.1877388
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5550735, 0.5489640
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9658536, 0.9748302
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0714855, 1.0517926

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472911, upper bound: 0.4354309
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4411237, upper bound: 0.4415946
time: 3.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6152351, 1.6129537
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7505862, 0.7590214
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8775880, 0.8855578
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2924459, 1.2710917
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1107011, 1.1139355
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8354765, 0.8530777
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1875401, 1.2047392
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5603242, 0.5433477
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9659499, 0.9718076
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0623965, 1.0605288

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4432446, upper bound: 0.4394739
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4370773, upper bound: 0.4456419
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6158454, 1.6123433
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7553272, 0.7542804
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8771317, 0.8860142
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2872145, 1.2763231
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1054313, 1.1192052
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8528455, 0.8357086
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1996562, 1.1926230
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5559839, 0.5476880
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9556626, 0.9820950
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0723605, 1.0505648

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472911, upper bound: 0.4354309
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4411237, upper bound: 0.4415946
time: 3.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6131251, 1.6164718
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7568847, 0.7553272
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8918836, 0.8771317
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2763231, 1.2930145
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1241283, 1.1070025
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8365282, 0.8528455
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1961770, 1.1996564
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5476880, 0.5596646
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9859711, 0.9556625
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0505648, 1.0766428

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4415943, upper bound: 0.4411249
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4354288, upper bound: 0.4472912
time: 3.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6137354, 1.6158614
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7616256, 0.7505863
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8914270, 0.8775880
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2710917, 1.2982459
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1188586, 1.1122723
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8538972, 0.8354765
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2082934, 1.1875401
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5433477, 0.5640050
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9756836, 0.9659498
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0605288, 1.0666788

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456417, upper bound: 0.4370772
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4394738, upper bound: 0.4432447
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6103899, 1.6184251
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7565018, 0.7555996
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8839872, 0.8827528
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2875397, 1.2772346
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1280339, 1.1015258
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8337847, 0.8548040
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1893275, 1.2045413
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5485983, 0.5583892
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9757813, 0.9629275
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0514400, 1.0754149

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4415943, upper bound: 0.4411247
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4354288, upper bound: 0.4472912
time: 3.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6110003, 1.6178148
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7612427, 0.7508585
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8835309, 0.8832092
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2823083, 1.2824659
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1227641, 1.1067955
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8511540, 0.8374350
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2014439, 1.1924249
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5442580, 0.5627295
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9654940, 0.9732147
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0614040, 1.0654509

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4456417, upper bound: 0.4370794
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4394738, upper bound: 0.4432446
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447306
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509037
time: 3.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6151063, 1.6138644
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7537532, 0.7559648
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8800886, 0.8853321
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2737648, 1.2943358
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1032910, 1.1229165
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8537849, 0.8355539
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2032530, 1.1909914
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5505110, 0.5535260
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9726824, 0.9679999
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0690057, 1.0542724

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420291, upper bound: 0.4406880
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4358643, upper bound: 0.4468562
time: 3.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6117609, 1.6164281
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7486295, 0.7609782
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8726487, 0.8904971
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2902133, 1.2733243
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1124663, 1.1121702
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8336724, 0.8548815
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1842873, 1.2079920
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5557616, 0.5479102
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9727801, 0.9649775
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0599165, 1.0630088

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447308
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509038
time: 3.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6123712, 1.6158178
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7533704, 0.7562373
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8721924, 0.8909535
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2849820, 1.2785556
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1071966, 1.1174400
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8510414, 0.8375125
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1964035, 1.1958758
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5514213, 0.5522505
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9624926, 0.9752648
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0698805, 1.0530448

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4420291, upper bound: 0.4406900
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4358643, upper bound: 0.4468563
time: 3.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4468561, upper bound: 0.4358655
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4406880, upper bound: 0.4420292
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4509035, upper bound: 0.4318201
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4447305, upper bound: 0.4379826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4468561, upper bound: 0.4358655
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4406880, upper bound: 0.4420292
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4509035, upper bound: 0.4318201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4447305, upper bound: 0.4379826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4432446, upper bound: 0.4394739
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4370773, upper bound: 0.4456419
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4472911, upper bound: 0.4354309
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4411237, upper bound: 0.4415946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4432446, upper bound: 0.4394739
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4370773, upper bound: 0.4456419
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4472911, upper bound: 0.4354309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4411237, upper bound: 0.4415946
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4415943, upper bound: 0.4411249
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4354288, upper bound: 0.4472912
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4456417, upper bound: 0.4370772
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4394738, upper bound: 0.4432447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4415943, upper bound: 0.4411247
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4354288, upper bound: 0.4472912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4456417, upper bound: 0.4370794
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4394738, upper bound: 0.4432446
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447306
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509037
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4420291, upper bound: 0.4406880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4358643, upper bound: 0.4468562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4379826, upper bound: 0.4447308
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4318179, upper bound: 0.4509038
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4420291, upper bound: 0.4406900
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.75
Output dim: 1, lower bound: -0.4358643, upper bound: 0.4468563

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6029956, 1.5966852
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7545021, 0.7481670
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8722806, 0.8427478
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1963201, 1.2222211
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1212673, 1.1070803
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8241510, 0.8392227
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1973138, 1.1938757
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5518923, 0.5542040
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9797180, 0.9632512
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0522985, 1.0735339

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468530, upper bound: 0.4352084
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400286, upper bound: 0.4352162
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6002872, 1.5993898
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7536379, 0.7490298
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8673801, 0.8476486
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2099977, 1.2085471
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1206763, 1.1076716
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8265133, 0.8368604
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1969028, 1.1942822
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5513519, 0.5547426
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9798992, 0.9630686
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0524139, 1.0734166

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400396, upper bound: 0.4352108
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400286, upper bound: 0.4420263
time: 3.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6036055, 1.5960751
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7592431, 0.7434261
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8718243, 0.8432044
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1910892, 1.2274525
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1159976, 1.1123501
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8415203, 0.8218536
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2094302, 1.1817595
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5475520, 0.5585444
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9694307, 0.9735385
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0622625, 1.0635699

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509004, upper bound: 0.4311619
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440760, upper bound: 0.4311696
time: 3.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6008976, 1.5987797
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7583790, 0.7442887
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8669238, 0.8481051
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2047658, 1.2137785
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1154063, 1.1129414
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8438821, 0.8194913
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2090192, 1.1821659
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5470116, 0.5590829
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9696119, 0.9733560
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0623779, 1.0634526

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440870, upper bound: 0.4311621
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440760, upper bound: 0.4379798
time: 3.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6002595, 1.5986392
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7541193, 0.7484394
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8643842, 0.8483692
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2075372, 1.2064445
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1251729, 1.1016043
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8214073, 0.8411813
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1904607, 1.1987602
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5528024, 0.5529280
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9695270, 0.9705163
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0531738, 1.0723058

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4468530, upper bound: 0.4352084
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400286, upper bound: 0.4352161
time: 3.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5975525, 1.6013439
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7532552, 0.7493023
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8594818, 0.8532699
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2212148, 1.1927667
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1245816, 1.1021954
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8237700, 0.8388190
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1900523, 1.1991665
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5522619, 0.5534669
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9697096, 0.9703337
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0532892, 1.0721887

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400396, upper bound: 0.4352108
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400286, upper bound: 0.4420263
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6008699, 1.5980289
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7588603, 0.7436985
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8639278, 0.8488257
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2023058, 1.2116759
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1199031, 1.1068741
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8387766, 0.8238122
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2025771, 1.1866438
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5484622, 0.5572683
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9592397, 0.9808036
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0631378, 1.0623418

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509004, upper bound: 0.4311619
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440760, upper bound: 0.4311696
time: 3.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5981629, 1.6007335
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7579963, 0.7445612
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8590255, 0.8537264
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2159834, 1.1979980
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1193120, 1.1074651
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8411388, 0.8214499
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.2021685, 1.1870502
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5479217, 0.5578072
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9594223, 0.9806211
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0632532, 1.0622247

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440870, upper bound: 0.4311643
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440760, upper bound: 0.4379798
time: 3.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.6043661, 1.5946884
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7466297, 0.7535454
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8609424, 0.8504922
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1989942, 1.2183111
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.1056998, 1.1177248
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8240390, 0.8393004
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1922736, 1.1973270
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5590556, 0.5437250
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9767168, 0.9653012
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0607755, 1.0611278

Time for backsubstitution: 12.65 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.7800273895263672
rel_dist={1: [-0.4509328816559992, 0.4509326440462238]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3104051
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3128611
time: 4.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.42 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 8.42
Output dim: 1, lower bound: -0.3104051, upper bound: 0.3104051
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.42
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

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104013, upper bound: 0.3103808
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3079224, upper bound: 0.3128620
time: 5.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.56 seconds
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 23.56
Output dim: 1, lower bound: -0.3104013, upper bound: 0.3103808
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.56
Output dim: 1, lower bound: -0.3079224, upper bound: 0.3128620

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4598646, 1.4606032
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6765385, 0.6814530
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8097591, 0.8178979
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1790283, 1.1781893
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0397627, 1.0466638
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7760313, 0.7769893
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1090198, 1.1131662
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5048585, 0.5019003
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8873924, 0.8854780
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9821820, 0.9772187

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078496, upper bound: 0.3128586
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078522, upper bound: 0.3127791
time: 3.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.27 seconds
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.27
Output dim: 1, lower bound: -0.3078496, upper bound: 0.3128586
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.27
Output dim: 1, lower bound: -0.3078522, upper bound: 0.3127791

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4592788, 1.4586499
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6764575, 0.6811806
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8080857, 0.8122766
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1678112, 1.1748626
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0358572, 1.0454967
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7754444, 0.7750309
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1075597, 1.1082815
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5039482, 0.5016278
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8852230, 0.8782135
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9813068, 0.9769576

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3128537
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3108429
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4579113, 1.4600172
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6762661, 0.6813719
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8041377, 0.8162256
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1757019, 1.1669726
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0385952, 1.0427585
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7740728, 0.7764026
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1041350, 1.1117066
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5045862, 0.5009900
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8801280, 0.8833091
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9819207, 0.9763436

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3127737
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3079140, upper bound: 0.3107619
time: 3.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.45 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.45
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3128537
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.45
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3108429
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.45
Output dim: 1, lower bound: -0.3058540, upper bound: 0.3127737
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.45
Output dim: 1, lower bound: -0.3079140, upper bound: 0.3107619

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4574788, 1.4571552
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6624514, 0.6695451
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8069663, 0.8109288
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1549675, 1.1594031
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0229144, 1.0299191
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7242688, 0.7325399
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0718257, 1.0786058
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4933114, 0.4888207
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8600043, 0.8478514
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9519939, 0.9526265

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3096875
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3128513
time: 3.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4561112, 1.4585226
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6622599, 0.6697364
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8030181, 0.8148779
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1628578, 1.1515131
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0256524, 1.0271807
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7228972, 0.7339118
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0684011, 1.0820310
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4939494, 0.4881830
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8549095, 0.8529469
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9526076, 0.9520125

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3096186
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3127738
time: 3.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.84 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.84
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3096875
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.84
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3128513
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.84
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3096186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.84
Output dim: 1, lower bound: -0.3026884, upper bound: 0.3127738

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4411666, 1.4421966
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6572480, 0.6647737
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7775216, 0.7839354
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0795712, 1.0771677
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0212276, 1.0285279
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7112684, 0.7183586
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0692978, 1.0762820
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4924127, 0.4881923
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8606718, 0.8484274
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9513059, 0.9518801

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3022527, upper bound: 0.3092445
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022473, upper bound: 0.3128493
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4397991, 1.4435644
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6570566, 0.6649650
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7735739, 0.7878836
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0874591, 1.0692778
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0239656, 1.0257894
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7098970, 0.7197305
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0658731, 1.0797086
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4930507, 0.4875546
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8555768, 0.8535229
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9519200, 0.9512664

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3023198, upper bound: 0.3091773
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3023166, upper bound: 0.3127620
time: 3.71 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.96 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 19.96
Output dim: 1, lower bound: -0.3022527, upper bound: 0.3092445
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.96
Output dim: 1, lower bound: -0.3022473, upper bound: 0.3128493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 19.96
Output dim: 1, lower bound: -0.3023198, upper bound: 0.3091773
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.96
Output dim: 1, lower bound: -0.3023166, upper bound: 0.3127620

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4347332, 1.4374359
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6572773, 0.6647733
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7736278, 0.7810558
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0739961, 1.0696273
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0218666, 1.0285254
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7093344, 0.7157434
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0677538, 1.0751387
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4924088, 0.4885721
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8597972, 0.8477789
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9509892, 0.9514517

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3117902
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3012043, upper bound: 0.3128470
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.4333656, 1.4388044
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.6570859, 0.6649648
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.7696795, 0.7850039
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.0818834, 1.0617373
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0246053, 1.0257871
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7079625, 0.7171153
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.0643291, 1.0785650
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.4930468, 0.4879341
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.8547022, 0.8528754
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9516032, 0.9508377

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4599

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3116996
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3127560
time: 3.88 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 20.63 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3117902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.63
Output dim: 1, lower bound: -0.3012043, upper bound: 0.3128470
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3116996
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.63
Output dim: 1, lower bound: -0.3012071, upper bound: 0.3127560

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1488
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 1500
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 899
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2476
type: RSZ, layer: 3, pos: 2487
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 1151
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 2144
type: RSZ, layer: 3, pos: 1188
type: RSZ, layer: 3, pos: 698

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2978885, upper bound: 0.3102405
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2988998, upper bound: 0.3090709
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1488
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 2321
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 1500
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 570
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 403
type: RSZ, layer: 3, pos: 899
type: RSZ, layer: 3, pos: 1432
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1971
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2476
type: RSZ, layer: 3, pos: 2487
type: RSZ, layer: 3, pos: 1984
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 1151
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 2879
type: RSZ, layer: 3, pos: 2144
type: RSZ, layer: 3, pos: 1188
type: RSZ, layer: 3, pos: 698

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2978885, upper bound: 0.3101448
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2989703, upper bound: 0.3089743
time: 4.03 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 20.96 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 20.96
Output dim: 1, lower bound: -0.2978885, upper bound: 0.3102405
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 20.96
Output dim: 1, lower bound: -0.2988998, upper bound: 0.3090709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 20.96
Output dim: 1, lower bound: -0.2978885, upper bound: 0.3101448
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 20.96
Output dim: 1, lower bound: -0.2989703, upper bound: 0.3089743
Binary search (step 2): status=Status.VERIFIED, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary search (step 3) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6193
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6193

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843160, upper bound: 0.3804870
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804845, upper bound: 0.3843157
time: 3.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.93
Output dim: 1, lower bound: -0.3843160, upper bound: 0.3804870
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.93
Output dim: 1, lower bound: -0.3804845, upper bound: 0.3843157

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5404811, 1.5378752
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7265694, 0.7251018
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8559086, 0.8522041
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2417686, 1.2400939
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0905890, 1.0919130
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8328493, 0.8314965
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1728534, 1.1704139
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5389273, 0.5355054
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9406836, 0.9458061
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0387025, 1.0368426

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843073, upper bound: 0.3770293
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808783, upper bound: 0.3804753
time: 3.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5378752, 1.5404811
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7251018, 0.7265694
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8522041, 0.8559085
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2400939, 1.2417684
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0919132, 1.0905890
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8314965, 0.8328495
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1704139, 1.1728534
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5355053, 0.5389272
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9458063, 0.9406835
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0368426, 1.0387025

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6113
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770319, upper bound: 0.3808785
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770319, upper bound: 0.3843074
time: 3.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 1, lower bound: -0.3843073, upper bound: 0.3770293
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 1, lower bound: -0.3808783, upper bound: 0.3804753
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 1, lower bound: -0.3770319, upper bound: 0.3808785
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.93
Output dim: 1, lower bound: -0.3770319, upper bound: 0.3843074

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5391102, 1.5380023
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7270951, 0.7197232
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8566685, 0.8444602
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2390950, 1.2403531
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0916202, 1.0812685
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8328559, 0.8314189
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1731827, 1.1669630
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5317639, 0.5362012
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9408842, 0.9437560
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0302258, 1.0376705

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843070, upper bound: 0.3770315
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3843070, upper bound: 0.3770315
time: 3.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5404811, 1.5365045
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7211908, 0.7251018
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8481646, 0.8522041
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2417686, 1.2374203
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0799446, 1.0919130
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8327720, 0.8314965
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1694024, 1.1704139
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5389273, 0.5283420
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9386333, 0.9458061
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0387025, 1.0283657

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808780, upper bound: 0.3804751
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808780, upper bound: 0.3804751
time: 3.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5365047, 1.5406082
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7256274, 0.7211908
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8529642, 0.8481647
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2374203, 1.2420275
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0929441, 1.0799446
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8315032, 0.8327719
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1707432, 1.1694025
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5283420, 0.5396231
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9460069, 0.9386333
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0283656, 1.0395304

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804755, upper bound: 0.3808777
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804755, upper bound: 0.3808778
time: 3.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5378752, 1.5391104
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7197232, 0.7265694
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8444602, 0.8559085
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2400939, 1.2390947
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0812685, 1.0905890
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8314188, 0.8328495
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1669629, 1.1728534
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5355053, 0.5317639
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9437560, 0.9406835
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0368426, 1.0302256

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6206
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770315, upper bound: 0.3843069
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770293, upper bound: 0.3843066
time: 3.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3843070, upper bound: 0.3770315
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3843070, upper bound: 0.3770315
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3808780, upper bound: 0.3804751
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3808780, upper bound: 0.3804751
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3804755, upper bound: 0.3808777
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3804755, upper bound: 0.3808778
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3770315, upper bound: 0.3843069
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.52
Output dim: 1, lower bound: -0.3770293, upper bound: 0.3843066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5392082, 1.5360489
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7271098, 0.7194507
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8569708, 0.8388388
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2278783, 1.2409718
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0877147, 1.0814703
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8329552, 0.8294604
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1734357, 1.1620783
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5308536, 0.5362480
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9412631, 0.9364915
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0293508, 1.0377162

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810816, upper bound: 0.3770223
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842978, upper bound: 0.3738686
time: 3.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5371573, 1.5380023
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7268227, 0.7197232
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8510473, 0.8444602
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2390950, 1.2291365
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0916202, 1.0773630
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8308977, 0.8314189
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1682978, 1.1669630
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5317639, 0.5352908
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9336199, 0.9437560
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0302258, 1.0367954

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810816, upper bound: 0.3770222
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842978, upper bound: 0.3738686
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5405791, 1.5345511
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7212054, 0.7248293
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8484671, 0.8465830
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2305510, 1.2380390
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0760391, 1.0921147
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8328711, 0.8295379
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1696556, 1.1655296
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5380169, 0.5283887
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9390122, 0.9385415
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0378275, 1.0284116

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777111, upper bound: 0.3804665
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808688, upper bound: 0.3772500
time: 3.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5385277, 1.5365045
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7209184, 0.7251018
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8425434, 0.8522041
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2417686, 1.2262037
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0799446, 1.0880077
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8308133, 0.8314965
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1645176, 1.1704139
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5389273, 0.5274317
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9313687, 0.9458061
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0387025, 1.0274909

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777111, upper bound: 0.3804665
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808688, upper bound: 0.3772500
time: 3.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5366027, 1.5386548
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7256422, 0.7209183
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8532648, 0.8425434
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2262037, 1.2426460
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0890388, 1.0801466
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8316022, 0.8308133
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1709955, 1.1645178
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5274317, 0.5396694
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9463848, 0.9313688
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0274909, 1.0395763

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772505, upper bound: 0.3808710
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804663, upper bound: 0.3777112
time: 3.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5345514, 1.5406082
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7253550, 0.7211908
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8473427, 0.8481647
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2374203, 1.2308109
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0929441, 1.0760391
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8295447, 0.8327719
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1658583, 1.1694025
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5283420, 0.5387129
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9387425, 0.9386333
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0283656, 1.0386555

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772505, upper bound: 0.3808710
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804663, upper bound: 0.3777112
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5379732, 1.5371571
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7197379, 0.7262969
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8447611, 0.8502874
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2288764, 1.2397132
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0773630, 1.0907912
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8315178, 0.8308908
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1672153, 1.1679691
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5345950, 0.5318102
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9441339, 0.9334189
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0359674, 1.0302715

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3738686, upper bound: 0.3842979
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770201, upper bound: 0.3810841
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5359218, 1.5391104
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7194508, 0.7265694
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8388388, 0.8559085
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2400939, 1.2278781
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0812685, 1.0866838
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.8294605, 0.8328495
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1620781, 1.1728534
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5355053, 0.5308536
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9364914, 0.9406835
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0368426, 1.0293508

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3738662, upper bound: 0.3842977
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770201, upper bound: 0.3810841
time: 3.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3810816, upper bound: 0.3770223
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3842978, upper bound: 0.3738686
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3810816, upper bound: 0.3770222
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3842978, upper bound: 0.3738686
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3777111, upper bound: 0.3804665
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3808688, upper bound: 0.3772500
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3777111, upper bound: 0.3804665
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3808688, upper bound: 0.3772500
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3772505, upper bound: 0.3808710
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3804663, upper bound: 0.3777112
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3772505, upper bound: 0.3808710
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3804663, upper bound: 0.3777112
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3738686, upper bound: 0.3842979
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3770201, upper bound: 0.3810841
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3738662, upper bound: 0.3842977
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.86
Output dim: 1, lower bound: -0.3770201, upper bound: 0.3810841

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5374086, 1.5347068
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7131037, 0.7090003
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8559654, 0.8374912
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2163422, 1.2255123
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0760894, 1.0658925
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7817799, 0.7913117
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1377020, 1.1354314
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5213019, 0.5234410
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9186162, 0.9061292
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0000377, 1.0158761

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810771, upper bound: 0.3722411
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3762332, upper bound: 0.3770137
time: 3.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5378664, 1.5342491
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7166593, 0.7054447
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8556230, 0.8378334
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2124188, 1.2294359
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0721371, 1.0698447
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7948066, 0.7782848
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1467891, 1.1263443
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5180466, 0.5266962
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9109010, 0.9138447
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0075107, 1.0084031

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842934, upper bound: 0.3690634
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3794493, upper bound: 0.3738607
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5353572, 1.5366602
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7128166, 0.7092727
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8500419, 0.8431122
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2275589, 1.2136769
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0799947, 1.0617855
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7797221, 0.7932701
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1325638, 1.1403164
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5222121, 0.5224839
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9109730, 0.9133942
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0009127, 1.0149553

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810771, upper bound: 0.3722411
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3762332, upper bound: 0.3770137
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5358150, 1.5362024
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7163724, 0.7057171
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8496995, 0.8434546
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2236354, 1.2176003
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0760424, 1.0657377
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7927488, 0.7802434
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1416512, 1.1312290
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5189569, 0.5257391
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9032575, 0.9211098
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0083857, 1.0074823

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842934, upper bound: 0.3690634
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3794493, upper bound: 0.3738607
time: 3.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5387790, 1.5332091
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7071992, 0.7143790
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8474615, 0.8452352
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2190154, 1.2225795
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0644138, 1.0765369
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7816955, 0.7913890
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1339216, 1.1388828
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5284652, 0.5155817
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9163655, 0.9081794
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0085144, 1.0065715

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
time: 3.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5392368, 1.5327513
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7107550, 0.7108232
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8471193, 0.8455775
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2150919, 1.2265031
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0604615, 1.0804892
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7947222, 0.7783623
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1430089, 1.1297957
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5252099, 0.5188370
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9086498, 0.9158949
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0159874, 0.9990985

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
time: 3.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5367277, 1.5351624
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7069123, 0.7146515
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8415380, 0.8508565
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2302325, 1.2107441
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0683193, 1.0724297
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7796379, 0.7933477
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1287837, 1.1437670
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5293756, 0.5146247
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9087220, 0.9154443
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0093894, 1.0056508

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
time: 3.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5371854, 1.5347047
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7104679, 0.7110957
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8411956, 0.8511989
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2263091, 1.2146678
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0643668, 1.0763822
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7926646, 0.7803209
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1378708, 1.1346799
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5261203, 0.5178800
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9010066, 0.9231597
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0168624, 0.9981778

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5348027, 1.5373127
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7116361, 0.7104681
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8522594, 0.8411956
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2146676, 1.2271864
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0774133, 1.0645690
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7804266, 0.7926646
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1352615, 1.1378709
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5178800, 0.5268625
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9237379, 0.9010066
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9981775, 1.0177362

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5352604, 1.5368550
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7151918, 0.7069123
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8519173, 0.8415380
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2107441, 1.2311101
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0734611, 1.0685213
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7934533, 0.7796378
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1443486, 1.1287838
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5146247, 0.5301176
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9160227, 0.9087220
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0056505, 1.0102632

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5327513, 1.5392661
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7113489, 0.7107403
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8463373, 0.8468168
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2258842, 1.2153513
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0813189, 1.0604615
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7783691, 0.7946231
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1301243, 1.1427559
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5187902, 0.5259058
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9160956, 0.9082716
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9990528, 1.0168154

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5332091, 1.5388083
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7149047, 0.7071847
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8459949, 0.8471590
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2219608, 1.2192750
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0773664, 1.0644138
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7913958, 0.7815963
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1392117, 1.1336685
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5155350, 0.5291611
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9083802, 0.9159871
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0065258, 1.0093422

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5361736, 1.5358150
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7057317, 0.7158465
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8437557, 0.8489397
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2173407, 1.2242537
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0657377, 1.0752132
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7803422, 0.7927420
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1314812, 1.1413223
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5250432, 0.5190033
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9214872, 0.9030567
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0066545, 1.0084314

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
time: 3.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5366313, 1.5353572
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7092875, 0.7122908
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8434134, 0.8492820
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2134173, 1.2281773
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0617855, 1.0791655
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7933692, 0.7797152
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1405685, 1.1322352
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5217880, 0.5222584
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9137715, 0.9107722
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0141275, 1.0009584

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5341222, 1.5377684
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7054447, 0.7161191
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8378334, 0.8545611
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2285578, 1.2124186
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0696430, 1.0711057
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7782849, 0.7947006
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1263442, 1.1462065
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5259535, 0.5180466
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9138447, 0.9103216
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0075295, 1.0075107

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
time: 3.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5345800, 1.5373106
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7090005, 0.7125633
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8374913, 0.8549033
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.2246344, 1.2163422
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0656910, 1.0750580
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7913116, 0.7816739
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1354313, 1.1371194
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5226984, 0.5213019
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9061292, 0.9180371
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0150025, 1.0000377

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5815
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5815

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770
time: 3.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3810771, upper bound: 0.3722411
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3762332, upper bound: 0.3770137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3842934, upper bound: 0.3690634
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3794493, upper bound: 0.3738607
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3810771, upper bound: 0.3722411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3762332, upper bound: 0.3770137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3842934, upper bound: 0.3690634
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3794493, upper bound: 0.3738607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.36
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5231273, 1.5183947
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7085483, 0.7037970
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8301964, 0.8080467
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1341071, 1.1535327
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0748458, 1.0642055
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7675986, 0.7789023
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1354821, 1.1329035
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5208086, 0.5225422
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9191923, 0.9068421
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9992914, 1.0152179

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810745, upper bound: 0.3717716
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757848, upper bound: 0.3717766
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5210965, 1.5204232
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7079003, 0.7044441
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8265209, 0.8117222
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1443648, 1.1432769
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0744026, 1.0646490
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7693701, 0.7771304
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1351738, 1.1332084
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5204031, 0.5229461
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9193282, 0.9067053
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -0.9993782, 1.0151299

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757888, upper bound: 0.3717738
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757831, upper bound: 0.3770107
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5235851, 1.5179369
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7121041, 0.7002413
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8298540, 0.8083889
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1301837, 1.1574562
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0708935, 1.0681578
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7806253, 0.7658753
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1445694, 1.1238164
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5175533, 0.5257975
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9114769, 0.9145577
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0067644, 1.0077449

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842907, upper bound: 0.3686088
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790010, upper bound: 0.3686156
time: 3.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5215542, 1.5199654
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7114559, 0.7008884
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8261786, 0.8120645
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1404414, 1.1472003
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0704501, 1.0686013
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7823968, 0.7641037
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1442611, 1.1241211
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5171479, 0.5262015
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9116127, 0.9144207
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0068512, 1.0076569

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790050, upper bound: 0.3686130
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3789993, upper bound: 0.3738569
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5210755, 1.5203488
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7082613, 0.7040695
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8242741, 0.8136680
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1453238, 1.1417000
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0787513, 1.0600985
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7655406, 0.7808609
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1303422, 1.1377879
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5217186, 0.5215852
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9115491, 0.9141072
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0001667, 1.0142969

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3810745, upper bound: 0.3717716
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757848, upper bound: 0.3717766
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5190451, 1.5223773
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7076132, 0.7047166
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8205972, 0.8173436
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1555820, 1.1314416
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0783079, 1.0605420
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7673125, 0.7790890
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1300359, 1.1380928
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5213132, 0.5219895
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9116859, 0.9139704
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0002532, 1.0142092

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757888, upper bound: 0.3717739
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3757831, upper bound: 0.3770105
time: 3.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5215333, 1.5198910
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7118170, 0.7005137
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8239317, 0.8140103
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1414003, 1.1456234
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0747991, 1.0640508
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7785678, 0.7678339
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1394296, 1.1287006
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5184634, 0.5248405
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9038336, 0.9218228
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0076396, 1.0068239

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3842907, upper bound: 0.3686088
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790010, upper bound: 0.3686156
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.5105915, -10.3324680, -12.5105915, -10.3324680, -1.5195029, 1.5219195
1: 3.1649604, 4.4064875, 3.1649604, 4.4064875, -0.7111690, 0.7011608
2: -4.9406466, -3.7702384, -4.9406466, -3.7702384, -0.8202548, 0.8176860
3: -12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.1516581, 1.1353652
4: -2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.0743556, 1.0644943
5: -10.0812330, -8.6300726, -10.0812330, -8.6300726, -0.7803392, 0.7660623
6: -8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.1391230, 1.1290056
7: -2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.5180581, 0.5252447
8: -3.7933545, -2.3891320, -3.7933545, -2.3891320, -0.9039705, 0.9216858
9: -12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.0077262, 1.0067362

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 4599

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3790050, upper bound: 0.3686130
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3789993, upper bound: 0.3738569
time: 3.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3810745, upper bound: 0.3717716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757848, upper bound: 0.3717766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757888, upper bound: 0.3717738
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757831, upper bound: 0.3770107
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3842907, upper bound: 0.3686088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3790010, upper bound: 0.3686156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3790050, upper bound: 0.3686130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3789993, upper bound: 0.3738569
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3810745, upper bound: 0.3717716
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757848, upper bound: 0.3717766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757888, upper bound: 0.3717739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3757831, upper bound: 0.3770105
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3842907, upper bound: 0.3686088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3790010, upper bound: 0.3686156
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3790050, upper bound: 0.3686130
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.63
Output dim: 1, lower bound: -0.3789993, upper bound: 0.3738569
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3777065, upper bound: 0.3756277
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3728940, upper bound: 0.3804611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3808638, upper bound: 0.3724133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3760751, upper bound: 0.3772458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3772457, upper bound: 0.3760773
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3724134, upper bound: 0.3808661
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3804615, upper bound: 0.3728939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3756280, upper bound: 0.3777062
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3738604, upper bound: 0.3794493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3690631, upper bound: 0.3842929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3770141, upper bound: 0.3762327
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 1, lower bound: -0.3722408, upper bound: 0.3810770
Binary search (step 3): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=0.7368426322937012
rel_dist={1: [-0.38431871261449, 0.38431867244677376]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1956.28 seconds
