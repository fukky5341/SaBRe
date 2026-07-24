## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.716269655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.6121473, 1.6121478)
1: (-6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7759919, 1.7759929)
2: (-8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6092749, 1.6092749)
3: (-4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4462447, 1.4462445)
4: (-7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2081947, 1.2081950)
5: (-5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3884630, 1.3884630)
6: (-13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5876408, 1.5876408)
7: (2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2234151, 1.2234151)
8: (-0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.3073392, 1.3073397)
9: (-8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4662395, 1.4662395)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.28 + 34.64 = 56.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.7198690, upper bound: 0.7198687

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6192
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7139732, upper bound: 0.7060259
time: 5.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7198665
time: 3.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.25 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 9.25
Output dim: 7, lower bound: -0.7139732, upper bound: 0.7060259
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.25
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7198665

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -15.3441124, -12.2636013, -15.3441200, -12.2636013, -1.5790257, 1.6121478
1: -6.8066788, -4.8410778, -6.8066797, -4.8410749, -1.7954149, 1.7759895
2: -8.3706245, -6.5761147, -8.3706236, -6.5761118, -1.6105318, 1.6092720
3: -4.6016631, -2.8533654, -4.6016669, -2.8533633, -1.4462419, 1.4584467
4: -7.5307083, -5.6608071, -7.5307088, -5.6608067, -1.2146811, 1.2081935
5: -5.9167213, -4.1329212, -5.9167237, -4.1329203, -1.3884611, 1.3954849
6: -13.9713306, -11.5294218, -13.9713326, -11.5294180, -1.5873899, 1.5712531
7: 2.7536249, 4.5407906, 2.7536235, 4.5407939, -1.2234147, 1.2108374
8: -0.9690351, 0.6157970, -0.9690433, 0.6157956, -1.2933931, 1.3073382
9: -8.3658695, -6.1971216, -8.3658676, -6.1971173, -1.4662380, 1.4428995

Time for backsubstitution: 20.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6192
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6192

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7060266, upper bound: 0.7139730
time: 3.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7060266, upper bound: 0.7198674
time: 3.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.59 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 27.59
Output dim: 7, lower bound: -0.7060266, upper bound: 0.7139730
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.59
Output dim: 7, lower bound: -0.7060266, upper bound: 0.7198674

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.3441124, -12.2636013, -15.3441124, -12.2636013, -1.5790253, 1.5790255
1: -6.8066788, -4.8410778, -6.8066788, -4.8410778, -1.7954140, 1.7954140
2: -8.3706245, -6.5761147, -8.3706245, -6.5761147, -1.6105289, 1.6105289
3: -4.6016631, -2.8533654, -4.6016631, -2.8533654, -1.4584432, 1.4584434
4: -7.5307083, -5.6608071, -7.5307083, -5.6608071, -1.2146807, 1.2146807
5: -5.9167213, -4.1329212, -5.9167213, -4.1329212, -1.3954835, 1.3954835
6: -13.9713306, -11.5294218, -13.9713306, -11.5294218, -1.5712528, 1.5712531
7: 2.7536249, 4.5407906, 2.7536249, 4.5407906, -1.2108369, 1.2108374
8: -0.9690351, 0.6157970, -0.9690351, 0.6157970, -1.2933931, 1.2933931
9: -8.3658695, -6.1971216, -8.3658695, -6.1971216, -1.4428997, 1.4428997

Time for backsubstitution: 21.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7046284, upper bound: 0.7198613
time: 3.85 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7060203, upper bound: 0.7198612
time: 3.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.36 seconds
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.7046284, upper bound: 0.7198613
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.36
Output dim: 7, lower bound: -0.7060203, upper bound: 0.7198612

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.3425970, -12.2677422, -15.3433895, -12.2656364, -1.5739784, 1.5707080
1: -6.7894754, -4.8416519, -6.7983980, -4.8413520, -1.7778444, 1.7864928
2: -8.3700514, -6.5811863, -8.3703499, -6.5785589, -1.6067877, 1.6037788
3: -4.6003027, -2.8570101, -4.6010199, -2.8551202, -1.4553204, 1.4542682
4: -7.5289493, -5.6647401, -7.5298848, -5.6627107, -1.2085917, 1.2060819
5: -5.9159307, -4.1336060, -5.9163504, -4.1332493, -1.3942671, 1.3941126
6: -13.9652882, -11.5296021, -13.9684181, -11.5295095, -1.5624733, 1.5668015
7: 2.7573862, 4.5396295, 2.7554364, 4.5402436, -1.2064710, 1.2079568
8: -0.9683785, 0.6151590, -0.9687204, 0.6154976, -1.2912145, 1.2901459
9: -8.3528957, -6.1973557, -8.3596201, -6.1972346, -1.4296808, 1.4363248

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196983
time: 3.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7128280, upper bound: 0.7198588
time: 3.65 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.3560381, -12.2549105, -15.3441095, -12.2636032, -1.5928202, 1.5857205
1: -6.8107786, -4.8157253, -6.8066473, -4.8410788, -1.7956295, 1.8219066
2: -8.3807907, -6.5731854, -8.3706245, -6.5761175, -1.6209183, 1.6124468
3: -4.6183910, -2.8516779, -4.6016617, -2.8533704, -1.4751759, 1.4601951
4: -7.5446405, -5.6602278, -7.5307055, -5.6608109, -1.2284083, 1.2150846
5: -5.9303808, -4.1303101, -5.9167185, -4.1329226, -1.4111414, 1.3997428
6: -13.9784670, -11.5187683, -13.9713202, -11.5294228, -1.5792952, 1.5818124
7: 2.7514663, 4.5525961, 2.7536311, 4.5407901, -1.2117596, 1.2236731
8: -0.9729834, 0.6230278, -0.9690347, 0.6157942, -1.2999864, 1.3019834
9: -8.3698311, -6.1787410, -8.3658447, -6.1971216, -1.4393039, 1.4532704

Time for backsubstitution: 20.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7196963
time: 5.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142650, upper bound: 0.7198566
time: 4.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.38 seconds
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196983
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -0.7128280, upper bound: 0.7198588
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7196963
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 7, lower bound: -0.7142650, upper bound: 0.7198566

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.3424158, -12.2680149, -15.3428154, -12.2664804, -1.5724220, 1.5692205
1: -6.7889385, -4.8441677, -6.7967114, -4.8492298, -1.7697649, 1.7829471
2: -8.3695574, -6.5814209, -8.3688011, -6.5793123, -1.6016827, 1.5982227
3: -4.5975628, -2.8573480, -4.5924273, -2.8561928, -1.4512925, 1.4451962
4: -7.5287752, -5.6680717, -7.5293369, -5.6731548, -1.1970594, 1.2016275
5: -5.9151616, -4.1339369, -5.9139400, -4.1343088, -1.3920069, 1.3907733
6: -13.9624310, -11.5299759, -13.9594612, -11.5307140, -1.5585537, 1.5573740
7: 2.7578225, 4.5382862, 2.7568107, 4.5360355, -1.2018464, 1.2056203
8: -0.9675474, 0.6150379, -0.9660707, 0.6151156, -1.2867665, 1.2845273
9: -8.3501511, -6.1976094, -8.3510084, -6.1980433, -1.4261756, 1.4276533

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178298
time: 4.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196976
time: 3.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.3425980, -12.2677450, -15.3467770, -12.2639675, -1.5784578, 1.5742569
1: -6.7894735, -4.8416619, -6.8113260, -4.8367853, -1.7832165, 1.8039970
2: -8.3700504, -6.5811863, -8.3744659, -6.5685811, -1.6147995, 1.6140614
3: -4.6002846, -2.8570113, -4.6053843, -2.8338866, -1.4763513, 1.4562778
4: -7.5289469, -5.6647596, -7.5656252, -5.6606469, -1.2097344, 1.2298601
5: -5.9159279, -4.1336079, -5.9202604, -4.1111245, -1.4142251, 1.3975995
6: -13.9652834, -11.5296040, -13.9726229, -11.4981546, -1.5855346, 1.5684829
7: 2.7573886, 4.5396242, 2.7434831, 4.5425539, -1.2085035, 1.2231009
8: -0.9683752, 0.6151586, -0.9798522, 0.6179156, -1.2914176, 1.3070731
9: -8.3528852, -6.1973553, -8.3646622, -6.1739354, -1.4527617, 1.4437308

Time for backsubstitution: 20.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7126696, upper bound: 0.7118752
time: 7.07 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7044059, upper bound: 0.7198569
time: 3.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.3558578, -12.2551823, -15.3435326, -12.2644510, -1.5912590, 1.5842321
1: -6.8102570, -4.8182430, -6.8049660, -4.8489571, -1.7875471, 1.8183608
2: -8.3802795, -6.5734200, -8.3690681, -6.5768714, -1.6158028, 1.6068792
3: -4.6156349, -2.8520093, -4.5930662, -2.8544405, -1.4710999, 1.4511247
4: -7.5444670, -5.6635580, -7.5301580, -5.6712546, -1.2168758, 1.2106311
5: -5.9296131, -4.1306424, -5.9143076, -4.1339865, -1.4088764, 1.3964014
6: -13.9756155, -11.5191450, -13.9623699, -11.5306320, -1.5753627, 1.5723753
7: 2.7518935, 4.5512486, 2.7549996, 4.5365806, -1.2071416, 1.2213354
8: -0.9721460, 0.6229062, -0.9663792, 0.6154127, -1.2955313, 1.2963829
9: -8.3670864, -6.1789989, -8.3572350, -6.1979356, -1.4357963, 1.4446514

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7178302
time: 4.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7057976
time: 7.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.3560371, -12.2549124, -15.3474998, -12.2619400, -1.5972996, 1.5892713
1: -6.8107772, -4.8157349, -6.8195810, -4.8365154, -1.8009987, 1.8365884
2: -8.3807898, -6.5731878, -8.3747330, -6.5661402, -1.6289325, 1.6227298
3: -4.6183748, -2.8516788, -4.6060185, -2.8321371, -1.4903369, 1.4622030
4: -7.5446401, -5.6602468, -7.5664635, -5.6587472, -1.2295513, 1.2364542
5: -5.9303780, -4.1303115, -5.9206343, -4.1107993, -1.4306045, 1.4032340
6: -13.9784603, -11.5187702, -13.9755259, -11.4980688, -1.6015520, 1.5834863
7: 2.7514682, 4.5525932, 2.7416706, 4.5430975, -1.2137911, 1.2365866
8: -0.9729805, 0.6230278, -0.9801660, 0.6182141, -1.3001776, 1.3189278
9: -8.3698215, -6.1787434, -8.3708878, -6.1738224, -1.4623871, 1.4609826

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7141066, upper bound: 0.7178267
time: 6.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7141066, upper bound: 0.7178298
time: 5.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 33.37 seconds
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178298
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196976
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7126696, upper bound: 0.7118752
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7044059, upper bound: 0.7198569
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7178302
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7122208, upper bound: 0.7057976
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7141066, upper bound: 0.7178267
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 33.37
Output dim: 7, lower bound: -0.7141066, upper bound: 0.7178298

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3428154, -12.2664804, -1.5716696, 1.5683987
1: -6.7877836, -4.8495293, -6.7967114, -4.8492298, -1.7690325, 1.7776842
2: -8.3685093, -6.5819416, -8.3688011, -6.5793123, -1.5987124, 1.5956950
3: -4.5917115, -2.8580873, -4.5924273, -2.8561928, -1.4453874, 1.4443469
4: -7.5284009, -5.6751847, -7.5293369, -5.6731548, -1.1964667, 1.1939578
5: -5.9135199, -4.1346598, -5.9139400, -4.1343088, -1.3900142, 1.3898640
6: -13.9563313, -11.5308027, -13.9594612, -11.5307140, -1.5523462, 1.5566764
7: 2.7587647, 4.5354218, 2.7568107, 4.5360355, -1.2011893, 1.2026784
8: -0.9657359, 0.6147766, -0.9660707, 0.6151156, -1.2835703, 1.2825041
9: -8.3442841, -6.1981640, -8.3510084, -6.1980433, -1.4204502, 1.4270968

Time for backsubstitution: 20.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7163984
time: 3.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178321
time: 3.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3428154, -12.2664804, -1.5763245, 1.5719063
1: -6.8023996, -4.8370848, -6.7967114, -4.8492298, -1.7876177, 1.7909393
2: -8.3741760, -6.5712104, -8.3688011, -6.5793123, -1.6037493, 1.6074333
3: -4.6046753, -2.8357792, -4.5924273, -2.8561928, -1.4582090, 1.4610848
4: -7.5646691, -5.6626768, -7.5293369, -5.6731548, -1.2175696, 1.2066035
5: -5.9198356, -4.1114826, -5.9139400, -4.1343088, -1.3969674, 1.4111476
6: -13.9694901, -11.4982491, -13.9594612, -11.5307140, -1.5648232, 1.5758536
7: 2.7454405, 4.5419421, 2.7568107, 4.5360355, -1.2172964, 1.2090991
8: -0.9795127, 0.6175756, -0.9660707, 0.6151156, -1.2968702, 1.2856917
9: -8.3579407, -6.1740561, -8.3510084, -6.1980433, -1.4388666, 1.4368985

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7182763
time: 3.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196976
time: 3.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3467770, -12.2639675, -1.5807371, 1.5774660
1: -6.8023996, -4.8370848, -6.8113260, -4.8367853, -1.7997875, 1.8084388
2: -8.3741760, -6.5712104, -8.3744659, -6.5685811, -1.6249204, 1.6219106
3: -4.6046753, -2.8357792, -4.6053843, -2.8338866, -1.4640570, 1.4630229
4: -7.5646691, -5.6626768, -7.5656252, -5.6606469, -1.2171488, 1.2146411
5: -5.9198356, -4.1114826, -5.9202604, -4.1111245, -1.4140334, 1.4138796
6: -13.9694901, -11.4982491, -13.9726229, -11.4981546, -1.5842643, 1.5883367
7: 2.7454405, 4.5419421, 2.7434831, 4.5425539, -1.2232239, 1.2247143
8: -0.9795127, 0.6175756, -0.9798522, 0.6179156, -1.3083549, 1.3098459
9: -8.3579407, -6.1740561, -8.3646622, -6.1739354, -1.4426231, 1.4492674

Time for backsubstitution: 20.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7163988
time: 3.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178325
time: 3.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.3554678, -12.2557526, -15.3435326, -12.2644510, -1.5904961, 1.5834091
1: -6.8091316, -4.8236084, -6.8049660, -4.8489571, -1.7868147, 1.8130903
2: -8.3791914, -6.5739355, -8.3690681, -6.5768714, -1.6128044, 1.6043444
3: -4.6097465, -2.8527339, -4.5930662, -2.8544405, -1.4650297, 1.4502702
4: -7.5440922, -5.6706729, -7.5301580, -5.6712546, -1.2162802, 1.2029614
5: -5.9279718, -4.1313696, -5.9143076, -4.1339865, -1.4068866, 1.3954816
6: -13.9695120, -11.5199776, -13.9623699, -11.5306320, -1.5691347, 1.5716836
7: 2.7528129, 4.5483732, 2.7549996, 4.5365806, -1.2064974, 1.2183852
8: -0.9703269, 0.6226459, -0.9663792, 0.6154127, -1.2923241, 1.2943602
9: -8.3612194, -6.1795549, -8.3572350, -6.1979356, -1.4300723, 1.4441037

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163945
time: 4.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163984
time: 3.42 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.3554678, -12.2557526, -15.3474998, -12.2619400, -1.5940037, 1.5880680
1: -6.8091316, -4.8236084, -6.8195810, -4.8365154, -1.8000689, 1.8287902
2: -8.3791914, -6.5739355, -8.3747330, -6.5661402, -1.6245494, 1.6093822
3: -4.6097465, -2.8527339, -4.6060185, -2.8321371, -1.4815359, 1.4631040
4: -7.5440922, -5.6706729, -7.5664635, -5.6587472, -1.2289283, 1.2251948
5: -5.9279718, -4.1313696, -5.9206343, -4.1107993, -1.4277043, 1.4024420
6: -13.9695120, -11.5199776, -13.9755259, -11.4980688, -1.5885458, 1.5812130
7: 2.7528129, 4.5483732, 2.7416706, 4.5430975, -1.2129149, 1.2323090
8: -0.9703269, 0.6226459, -0.9801660, 0.6182141, -1.2954907, 1.3076544
9: -8.3612194, -6.1795549, -8.3708878, -6.1738224, -1.4539833, 1.4601121

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163976
time: 3.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163980
time: 3.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.3594093, -12.2532406, -15.3474998, -12.2619400, -1.5995770, 1.5924726
1: -6.8236847, -4.8111382, -6.8195810, -4.8365154, -1.8175488, 1.8420110
2: -8.3849325, -6.5632133, -8.3747330, -6.5661402, -1.6389880, 1.6305680
3: -4.6227713, -2.8304629, -4.6060185, -2.8321371, -1.4836969, 1.4688737
4: -7.5803432, -5.6581745, -7.5664635, -5.6587472, -1.2369971, 1.2236443
5: -5.9342909, -4.1082211, -5.9206343, -4.1107993, -1.4309034, 1.4194474
6: -13.9826975, -11.4874382, -13.9755259, -11.4980688, -1.6010957, 1.5950725
7: 2.7395205, 4.5549111, 2.7416706, 4.5430975, -1.2285001, 1.2387464
8: -0.9840770, 0.6254587, -0.9801660, 0.6182141, -1.3145270, 1.3216991
9: -8.3748693, -6.1554675, -8.3708878, -6.1738224, -1.4522591, 1.4648809

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163961
time: 3.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163954
time: 4.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 29.07 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7163984
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178321
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7182763
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7196976
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7163988
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7107836, upper bound: 0.7178325
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163945
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163984
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163976
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163980
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122207, upper bound: 0.7163961
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.07
Output dim: 7, lower bound: -0.7122209, upper bound: 0.7163954

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3420238, -12.2685862, -1.5676699, 1.5676701
1: -6.7877836, -4.8495293, -6.7877836, -4.8495293, -1.7687168, 1.7687173
2: -8.3685093, -6.5819416, -8.3685093, -6.5819416, -1.5953650, 1.5953650
3: -4.5917115, -2.8580873, -4.5917115, -2.8580873, -1.4435787, 1.4435787
4: -7.5284009, -5.6751847, -7.5284009, -5.6751847, -1.1926527, 1.1926527
5: -5.9135199, -4.1346598, -5.9135199, -4.1346598, -1.3894777, 1.3894780
6: -13.9563313, -11.5308027, -13.9563313, -11.5308027, -1.5521874, 1.5521877
7: 2.7587647, 4.5354218, 2.7587647, 4.5354218, -1.2006581, 1.2006581
8: -0.9657359, 0.6147766, -0.9657359, 0.6147766, -1.2820730, 1.2820730
9: -8.3442841, -6.1981640, -8.3442841, -6.1981640, -1.4203138, 1.4203136

Time for backsubstitution: 20.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7094099, upper bound: 0.7123503
time: 3.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107843, upper bound: 0.7163972
time: 6.20 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3554678, -12.2557526, -1.5771208, 1.5828507
1: -6.7877836, -4.8495293, -6.8091316, -4.8236084, -1.7917261, 1.7894974
2: -8.3685093, -6.5819416, -8.3791914, -6.5739355, -1.6033549, 1.6063747
3: -4.5917115, -2.8580873, -4.6097465, -2.8527339, -1.4496450, 1.4615414
4: -7.5284009, -5.6751847, -7.5440922, -5.6706729, -1.1971388, 1.2089272
5: -5.9135199, -4.1346598, -5.9279718, -4.1313696, -1.3945050, 1.4058702
6: -13.9563313, -11.5308027, -13.9695120, -11.5199776, -1.5630317, 1.5657258
7: 2.7587647, 4.5354218, 2.7528129, 4.5483732, -1.2144926, 1.2061291
8: -0.9657359, 0.6147766, -0.9703269, 0.6226459, -1.2906113, 1.2894716
9: -8.3442841, -6.1981640, -8.3612194, -6.1795549, -1.4387808, 1.4365406

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7094099, upper bound: 0.7137613
time: 3.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107843, upper bound: 0.7178302
time: 6.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3420238, -12.2685862, -1.5723252, 1.5711777
1: -6.8023996, -4.8370848, -6.7877836, -4.8495293, -1.7873030, 1.7819724
2: -8.3741760, -6.5712104, -8.3685093, -6.5819416, -1.6004024, 1.6071033
3: -4.6046753, -2.8357792, -4.5917115, -2.8580873, -1.4564004, 1.4603195
4: -7.5646691, -5.6626768, -7.5284009, -5.6751847, -1.2147861, 1.2052984
5: -5.9198356, -4.1114826, -5.9135199, -4.1346598, -1.3964310, 1.4107618
6: -13.9694901, -11.4982491, -13.9563313, -11.5308027, -1.5646648, 1.5723915
7: 2.7454405, 4.5419421, 2.7587647, 4.5354218, -1.2167652, 1.2070787
8: -0.9795127, 0.6175756, -0.9657359, 0.6147766, -1.2953730, 1.2852612
9: -8.3579407, -6.1740561, -8.3442841, -6.1981640, -1.4387302, 1.4318423

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7094095, upper bound: 0.7142155
time: 3.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107839, upper bound: 0.7182744
time: 5.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3554678, -12.2557526, -1.5817757, 1.5863583
1: -6.8023996, -4.8370848, -6.8091316, -4.8236084, -1.7983432, 1.8027520
2: -8.3741760, -6.5712104, -8.3791914, -6.5739355, -1.6083922, 1.6181130
3: -4.6046753, -2.8357792, -4.6097465, -2.8527339, -1.4624667, 1.4697592
4: -7.5646691, -5.6626768, -7.5440922, -5.6706729, -1.2169418, 1.2215731
5: -5.9198356, -4.1114826, -5.9279718, -4.1313696, -1.4014583, 1.4219294
6: -13.9694901, -11.4982491, -13.9695120, -11.5199776, -1.5725608, 1.5835655
7: 2.7454405, 4.5419421, 2.7528129, 4.5483732, -1.2305999, 1.2125497
8: -0.9795127, 0.6175756, -0.9703269, 0.6226459, -1.3039112, 1.2926598
9: -8.3579407, -6.1740561, -8.3612194, -6.1795549, -1.4571967, 1.4408989

Time for backsubstitution: 21.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 451
type: A, layer: 1, pos: 46

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 451

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7094095, upper bound: 0.7156132
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7107839, upper bound: 0.7196955
time: 5.88 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.92 + 548.51 = 605.43 seconds
