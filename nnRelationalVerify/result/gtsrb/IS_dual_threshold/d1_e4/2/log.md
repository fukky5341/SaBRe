## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 12.5603180091


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.9000664)
1: (-3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5087318)
2: (-0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4614372)
3: (-1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0319729)
4: (-11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6887894, 14.6887856)
5: (1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796)
6: (-39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2155037, 15.2154999)
7: (-3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6391640)
8: (-6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1128178)
9: (-4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0374336)
10: (1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9535446)
11: (-11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476)
12: (-11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0364494, 15.0364494)
13: (-18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293907, 16.6293945)
14: (4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7749405)
15: (-8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198)
16: (-16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8331184)
17: (6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2550125)
18: (-14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4317722, 14.4317741)
19: (-20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5550842, 14.5550919)
20: (-2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6365433, 12.6365433)
21: (-11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549)
22: (-3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717941, 14.9717941)
23: (-14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3402557)
24: (-19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2769775, 9.2769775)
25: (-5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234940)
26: (-21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3724327, 19.3724365)
27: (-16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2311211, 13.2311211)
28: (-12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141)
29: (-5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9777222, 14.9777222)
30: (-10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620842)
31: (-10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6624298, 14.6624260)
32: (-24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3205376, 13.3205414)
33: (-69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6724014, 16.6724014)
34: (-53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1694450, 14.1694450)
35: (-47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0219498, 13.0219536)
36: (-42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1241531, 15.1241570)
37: (-86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9252739, 18.9252701)
38: (-52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3915100, 18.3915100)
39: (-76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0968819, 16.0968819)
40: (-67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520813, 14.3520813)
41: (-55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7036438, 16.7036476)
42: (-29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2716675)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.74 + 20.69 = 23.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -12.5728910, upper bound: 12.5728909

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 887
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 887

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5378280, upper bound: 12.5710522
time: 13.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5718465, upper bound: 12.5718468
time: 7.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.53
Output dim: 14, lower bound: -12.5378280, upper bound: 12.5710522
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.53
Output dim: 14, lower bound: -12.5718465, upper bound: 12.5718468

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.1235180, 3.6745415, -12.1244726, 3.6795990, -13.8921051, 13.8873711
1: -3.6717880, 7.3950539, -3.6721368, 7.3997259, -8.5028076, 8.4982452
2: -0.7454842, 13.4278774, -0.7465404, 13.4355831, -13.4512253, 13.4445000
3: -1.1316417, 11.3070602, -1.1323649, 11.3137321, -12.0232391, 12.0167580
4: -11.1149940, 5.4829149, -11.1158676, 5.4885273, -14.6810837, 14.6763306
5: 1.8399134, 17.7428169, 1.8389769, 17.7499828, -15.9100695, 15.9038401
6: -39.9336357, -18.2172108, -39.9348335, -18.2115116, -15.2057114, 15.2026939
7: -3.5818722, 12.2489910, -3.5828848, 12.2592869, -13.6256256, 13.6154823
8: -6.7099609, 8.5672398, -6.7112551, 8.5729942, -12.1034355, 12.0980644
9: -4.7909579, 11.7185555, -4.7954302, 11.7203045, -13.0222397, 13.0263519
10: 1.3032289, 25.7432251, 1.2970800, 25.7445107, -20.9365005, 20.9400253
11: -11.5066128, 4.2876549, -11.5102129, 4.2884569, -15.7950697, 15.7978678
12: -11.8989515, 9.8293657, -11.9095402, 9.8306103, -15.0126190, 15.0230370
13: -18.5693626, 6.7317352, -18.5751896, 6.7326612, -16.6182632, 16.6167450
14: 4.9476433, 36.4224358, 4.9338331, 36.4228096, -26.7413254, 26.7550430
15: -8.7045717, 9.2922878, -8.7096100, 9.2941685, -17.9987411, 18.0018978
16: -16.7459984, 2.5325413, -16.7480564, 2.5407639, -14.8193588, 14.8131638
17: 6.2069449, 30.6587372, 6.1948967, 30.6595421, -17.2266769, 17.2387085
18: -14.3964405, 5.1253347, -14.3984671, 5.1306496, -14.4200974, 14.4168739
19: -20.2774334, -4.3149395, -20.2813034, -4.3148332, -14.5439034, 14.5486755
20: -2.4251988, 11.2284622, -2.4276145, 11.2292681, -12.6295662, 12.6311264
21: -11.0751953, 3.2525582, -11.0808163, 3.2529693, -14.3281651, 14.3333740
22: -3.6891873, 13.1175413, -3.6996503, 13.1184635, -14.9479713, 14.9580574
23: -14.5816708, 0.3542733, -14.5846739, 0.3546395, -14.3289871, 14.3332100
24: -19.9386520, -5.1106939, -19.9407349, -5.1101780, -9.2698059, 9.2727547
25: -5.4521093, 10.8649063, -5.4604802, 10.8653002, -13.8052902, 13.8133812
26: -21.0028458, 1.2161527, -21.0162735, 1.2172148, -19.3432007, 19.3558502
27: -16.0113297, 2.1817508, -16.0123291, 2.1862812, -13.2207680, 13.2201080
28: -12.7956953, 4.6509962, -12.7994118, 4.6516528, -17.4473476, 17.4504089
29: -5.5840931, 11.8934250, -5.5946465, 11.8940754, -14.9553452, 14.9652557
30: -10.0517769, 6.2082453, -10.0565042, 6.2089372, -13.5512886, 13.5553474
31: -10.9790907, 6.9534969, -10.9814892, 6.9554696, -14.6516914, 14.6536827
32: -24.9259033, -4.5553112, -24.9270706, -4.5509624, -13.3125725, 13.3107605
33: -69.3157349, -40.0927124, -69.3164139, -40.0889969, -16.6669846, 16.6655693
34: -53.7644577, -30.8989429, -53.7650604, -30.8943882, -14.1602516, 14.1588783
35: -47.8217697, -26.0580750, -47.8232117, -26.0564823, -13.0113983, 13.0143890
36: -42.8225861, -19.2635136, -42.8245926, -19.2623100, -15.1150131, 15.1172485
37: -86.6762772, -55.5350342, -86.6789398, -55.5321579, -18.9111900, 18.9154358
38: -52.9514236, -24.3167667, -52.9531784, -24.3143482, -18.3829613, 18.3837280
39: -76.5608749, -44.6169052, -76.5624313, -44.6154251, -16.0841599, 16.0875702
40: -67.2545624, -43.5287018, -67.2555542, -43.5175171, -14.3438606, 14.3403282
41: -55.4313126, -32.9514008, -55.4319839, -32.9441109, -16.6933708, 16.6922836
42: -29.4713097, -9.8708220, -29.4722290, -9.8690462, -17.2601891, 17.2622414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 689

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5374199, upper bound: 12.5540603
time: 7.33 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5375416, upper bound: 12.5707519
time: 13.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.1424036, 3.6834550, -12.1250944, 3.6832433, -13.9182663, 13.8955727
1: -3.6844096, 7.4039040, -3.6723666, 7.4031839, -8.5144997, 8.5046692
2: -0.7707534, 13.4408360, -0.7473167, 13.4408321, -13.4820595, 13.4559288
3: -1.1444782, 11.3201113, -1.1326797, 11.3192596, -12.0471230, 12.0259781
4: -11.1286983, 5.4934554, -11.1163445, 5.4923449, -14.6985474, 14.6860809
5: 1.8208046, 17.7548256, 1.8383989, 17.7549286, -15.9341240, 15.9164267
6: -39.9383049, -18.2168655, -39.9352036, -18.2138615, -15.2003937, 15.2096329
7: -3.6143732, 12.2669392, -3.5833342, 12.2666016, -13.6740646, 13.6294060
8: -6.7293844, 8.5767374, -6.7120166, 8.5768738, -12.1282272, 12.1036549
9: -4.7936816, 11.7262383, -4.7945309, 11.7213850, -13.0295029, 13.0310822
10: 1.2893009, 25.7515926, 1.2921867, 25.7456474, -20.9481659, 20.9514618
11: -11.5132885, 4.2892675, -11.5122166, 4.2887440, -15.8020325, 15.8014841
12: -11.9159145, 9.8601780, -11.9167747, 9.8314381, -15.0286407, 15.0685120
13: -18.5598774, 6.7426496, -18.5674744, 6.7328806, -16.6419754, 16.5835342
14: 4.9214716, 36.4357986, 4.9255276, 36.4229965, -26.7673798, 26.7663345
15: -8.7022581, 9.2965107, -8.7063417, 9.2949247, -17.9971828, 18.0028534
16: -16.7690849, 2.5494692, -16.7496662, 2.5483115, -14.8303185, 14.8277092
17: 6.1836972, 30.6795158, 6.1861854, 30.6600952, -17.2507324, 17.2628250
18: -14.4069605, 5.1344957, -14.3996086, 5.1338615, -14.4268513, 14.4259644
19: -20.2857704, -4.3159223, -20.2827797, -4.3148746, -14.5546494, 14.5578728
20: -2.4314206, 11.2301636, -2.4292457, 11.2289076, -12.6359634, 12.6363411
21: -11.0896959, 3.2664104, -11.0855141, 3.2532258, -14.3429222, 14.3519249
22: -3.7089071, 13.1503639, -3.7073865, 13.1188211, -14.9634933, 15.0015945
23: -14.5901575, 0.3550272, -14.5871525, 0.3546495, -14.3327637, 14.3452187
24: -19.9415817, -5.1080761, -19.9404716, -5.1099877, -9.2763786, 9.2824707
25: -5.4717484, 10.8899231, -5.4683170, 10.8653584, -13.8187714, 13.8469009
26: -21.0277443, 1.2598844, -21.0268192, 1.2179654, -19.3606491, 19.4082413
27: -16.0273056, 2.1886740, -16.0129604, 2.1888900, -13.2198105, 13.2310028
28: -12.8048458, 4.6535411, -12.8020267, 4.6519461, -17.4567909, 17.4555683
29: -5.6023922, 11.9284229, -5.6019192, 11.8943615, -14.9705124, 15.0081749
30: -10.0607548, 6.2242012, -10.0595121, 6.2091522, -13.5595856, 13.5745125
31: -10.9917336, 6.9552627, -10.9829731, 6.9561453, -14.6584167, 14.6585197
32: -24.9318714, -4.5509596, -24.9274197, -4.5506601, -13.3139076, 13.3179932
33: -69.3245468, -40.0833397, -69.3167725, -40.0864143, -16.6696167, 16.6745377
34: -53.7692184, -30.8868942, -53.7654953, -30.8901787, -14.1605110, 14.1711311
35: -47.8119736, -26.0606766, -47.8158875, -26.0554810, -13.0132599, 13.0197029
36: -42.8111534, -19.2623959, -42.8163071, -19.2616386, -15.1160011, 15.1193733
37: -86.6755219, -55.5312424, -86.6755829, -55.5307312, -18.9228287, 18.9336815
38: -52.9568634, -24.3115883, -52.9543610, -24.3127289, -18.3899498, 18.3905716
39: -76.5538635, -44.6156006, -76.5559311, -44.6147766, -16.0890312, 16.0968323
40: -67.2899628, -43.5070877, -67.2557526, -43.5072746, -14.3306770, 14.3518867
41: -55.4452553, -32.9358940, -55.4324188, -32.9381332, -16.6834679, 16.7103195
42: -29.4688549, -9.8823919, -29.4724579, -9.8775654, -17.2435608, 17.2826691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 689

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5714234, upper bound: 12.5548618
time: 16.52 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5715464, upper bound: 12.5715466
time: 14.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 33.76 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 33.76
Output dim: 14, lower bound: -12.5374199, upper bound: 12.5540603
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 33.76
Output dim: 14, lower bound: -12.5375416, upper bound: 12.5707519
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 33.76
Output dim: 14, lower bound: -12.5714234, upper bound: 12.5548618
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 33.76
Output dim: 14, lower bound: -12.5715464, upper bound: 12.5715466

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -12.1228676, 3.6735325, -12.1240807, 3.6790042, -13.8908348, 13.8825912
1: -3.6707006, 7.3946285, -3.6715050, 7.3994842, -8.4936104, 8.4964981
2: -0.7434466, 13.4274893, -0.7452989, 13.4353294, -13.4485321, 13.4438057
3: -1.1293608, 11.3064957, -1.1309255, 11.3134117, -12.0310593, 12.0112419
4: -11.1139221, 5.4821835, -11.1152382, 5.4881001, -14.6693420, 14.6748466
5: 1.8429351, 17.7423534, 1.8409395, 17.7497368, -15.9068012, 15.9014139
6: -39.9310455, -18.2173462, -39.9333115, -18.2115936, -15.1346626, 15.2009888
7: -3.5801656, 12.2485895, -3.5818055, 12.2590685, -13.6045303, 13.6128197
8: -6.7060843, 8.5668497, -6.7089944, 8.5727844, -12.1015396, 12.0950317
9: -4.7903290, 11.7169704, -4.7950792, 11.7193804, -13.0208206, 13.0003242
10: 1.3076134, 25.7424870, 1.2996078, 25.7441006, -20.9281845, 20.9470673
11: -11.5043974, 4.2873330, -11.5089188, 4.2882543, -15.7926521, 15.7962513
12: -11.8981810, 9.8286266, -11.9090958, 9.8301353, -15.0023117, 15.0218201
13: -18.5691071, 6.7269239, -18.5750446, 6.7299328, -16.6100426, 16.6129875
14: 4.9489994, 36.4209785, 4.9346590, 36.4219666, -26.7390442, 26.7267838
15: -8.7042027, 9.2897081, -8.7094021, 9.2926598, -17.9968624, 17.9991112
16: -16.7398949, 2.5321541, -16.7444458, 2.5405104, -14.8179703, 14.8092957
17: 6.2082224, 30.6578655, 6.1956630, 30.6590385, -17.2250519, 17.2361107
18: -14.3949270, 5.1247096, -14.3975811, 5.1302896, -14.4147186, 14.4153137
19: -20.2765236, -4.3189349, -20.2807560, -4.3172016, -14.5412369, 14.5421333
20: -2.4240568, 11.2281303, -2.4269545, 11.2290821, -12.6226692, 12.6292648
21: -11.0739689, 3.2522469, -11.0800838, 3.2527816, -14.3267508, 14.3323307
22: -3.6887331, 13.1151867, -3.6994095, 13.1171074, -14.9461403, 14.9437752
23: -14.5807343, 0.3526957, -14.5841227, 0.3537421, -14.3273239, 14.3085060
24: -19.9380112, -5.1128082, -19.9403095, -5.1114001, -9.2651100, 9.2752571
25: -5.4512444, 10.8626595, -5.4599562, 10.8637676, -13.8023262, 13.8018303
26: -21.0021763, 1.2133939, -21.0158367, 1.2153800, -19.3409195, 19.3159561
27: -16.0108032, 2.1803708, -16.0120163, 2.1854112, -13.2249069, 13.2172470
28: -12.7949238, 4.6494164, -12.7989235, 4.6507325, -17.4456558, 17.4483395
29: -5.5836563, 11.8916569, -5.5943899, 11.8930588, -14.9538651, 14.9434433
30: -10.0478277, 6.2077827, -10.0542297, 6.2086706, -13.5503464, 13.5526276
31: -10.9772682, 6.9533324, -10.9804096, 6.9553699, -14.6488266, 14.6498528
32: -24.9247055, -4.5558014, -24.9263763, -4.5512447, -13.2828331, 13.3070145
33: -69.3139191, -40.0936928, -69.3153534, -40.0895691, -16.6379967, 16.6635742
34: -53.7633133, -30.9000702, -53.7643890, -30.8950462, -14.1195602, 14.1514244
35: -47.8212242, -26.0588245, -47.8229294, -26.0569534, -12.9964218, 13.0076065
36: -42.8222122, -19.2724743, -42.8244095, -19.2674332, -15.1113434, 15.1081276
37: -86.6754837, -55.5419960, -86.6784668, -55.5362968, -18.9058495, 18.9226303
38: -52.9496422, -24.3179245, -52.9521561, -24.3150177, -18.3471489, 18.3814011
39: -76.5595474, -44.6179390, -76.5616150, -44.6160126, -16.0814247, 16.0869789
40: -67.2527237, -43.5288467, -67.2543793, -43.5176010, -14.3360748, 14.3391666
41: -55.4304657, -32.9521561, -55.4315262, -32.9445190, -16.6790543, 16.6868820
42: -29.4701424, -9.8710518, -29.4715157, -9.8692036, -17.2568703, 17.2574310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 889

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5359199, upper bound: 12.5527998
time: 7.89 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5367353, upper bound: 12.5696245
time: 6.84 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -12.1187115, 3.6596818, -12.1193609, 3.6695285, -13.8803635, 13.8658371
1: -3.6666403, 7.3932467, -3.6619053, 7.3996181, -8.4949493, 8.4868813
2: -0.7555866, 13.4337196, -0.7396356, 13.4366188, -13.4608536, 13.4378014
3: -1.1276621, 11.3093395, -1.1229806, 11.3139868, -12.0270309, 12.0128059
4: -11.0975513, 5.4682760, -11.0985317, 5.4875603, -14.6620178, 14.6445847
5: 1.8417869, 17.7494469, 1.8502731, 17.7522869, -15.9104996, 15.8991737
6: -39.8616447, -18.2589645, -39.8910599, -18.2143402, -15.1246490, 15.1227112
7: -3.5775714, 12.2498531, -3.5634809, 12.2638073, -13.6387482, 13.5979042
8: -6.7206497, 8.5668621, -6.7088175, 8.5715542, -12.1138954, 12.0904522
9: -4.7604470, 11.6880684, -4.7879138, 11.7004204, -12.9763641, 12.9877510
10: 1.3231330, 25.7409306, 1.3102102, 25.7403374, -20.9048767, 20.9075089
11: -11.4948463, 4.2867470, -11.5039778, 4.2869415, -15.7817879, 15.7907248
12: -11.8959084, 9.8422527, -11.9062481, 9.8290501, -15.0063972, 15.0395699
13: -18.5550690, 6.7085838, -18.5659847, 6.7154551, -16.6230545, 16.5505905
14: 4.9557447, 36.3976669, 4.9299631, 36.4018402, -26.7113953, 26.7239227
15: -8.6658030, 9.2265930, -8.7035465, 9.2550735, -17.9208755, 17.9301395
16: -16.7361870, 2.5490961, -16.7333565, 2.5468965, -14.7923279, 14.8115959
17: 6.2099605, 30.6526604, 6.1890354, 30.6458397, -17.2090721, 17.2332726
18: -14.3681860, 5.1177626, -14.3776455, 5.1284723, -14.3798847, 14.3849621
19: -20.2659721, -4.3322964, -20.2741280, -4.3245192, -14.5231133, 14.5308228
20: -2.4107358, 11.2179546, -2.4181426, 11.2257576, -12.6145554, 12.6162109
21: -11.0681143, 3.2607379, -11.0737505, 3.2521429, -14.3202572, 14.3344879
22: -3.6924758, 13.1067505, -3.7041407, 13.0941582, -14.9179878, 14.9535103
23: -14.5572138, 0.3105845, -14.5816755, 0.3287392, -14.2759094, 14.2953453
24: -19.9359322, -5.1200886, -19.9383106, -5.1162391, -9.2687111, 9.2670593
25: -5.4580379, 10.8595181, -5.4649239, 10.8490925, -13.7882156, 13.8121071
26: -21.0019360, 1.2035255, -21.0233097, 1.1853557, -19.3112106, 19.3497314
27: -16.0095329, 2.1754837, -16.0069275, 2.1816130, -13.1890793, 13.2126541
28: -12.7725773, 4.6099396, -12.7970810, 4.6265535, -17.3991318, 17.4070206
29: -5.5666809, 11.8783197, -5.5993080, 11.8656979, -14.9070206, 14.9552650
30: -10.0476885, 6.2170534, -10.0531816, 6.2042561, -13.5386734, 13.5580826
31: -10.9581642, 6.9536428, -10.9656334, 6.9553595, -14.6233101, 14.6390343
32: -24.8966599, -4.5687504, -24.9076023, -4.5515881, -13.2831802, 13.2823410
33: -69.2893372, -40.1176071, -69.2964783, -40.0927277, -16.6287308, 16.6214142
34: -53.7341614, -30.9101219, -53.7445526, -30.8939342, -14.1234627, 14.1226006
35: -47.8030777, -26.0722961, -47.8108978, -26.0584335, -13.0002937, 12.9960785
36: -42.8110237, -19.2767162, -42.8121338, -19.2683315, -15.1032524, 15.0994148
37: -86.6664886, -55.5498924, -86.6699066, -55.5408516, -18.9105263, 18.9134178
38: -52.9092903, -24.3454399, -52.9279022, -24.3158150, -18.3399544, 18.3299255
39: -76.5241241, -44.6384659, -76.5398254, -44.6188889, -16.0541840, 16.0538826
40: -67.2487335, -43.5147095, -67.2354279, -43.5078201, -14.2844620, 14.3166504
41: -55.4209824, -32.9437256, -55.4199677, -32.9395332, -16.6602478, 16.6865501
42: -29.4538918, -9.8851166, -29.4649792, -9.8787584, -17.2232819, 17.2665672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 889

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5693603, upper bound: 12.5372047
time: 7.74 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5703000, upper bound: 12.5537152
time: 28.90 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -12.1417370, 3.6824403, -12.1247063, 3.6826513, -13.9169731, 13.8907890
1: -3.6833234, 7.4034958, -3.6717460, 7.4029226, -8.5052986, 8.5029430
2: -0.7687027, 13.4404478, -0.7461023, 13.4406176, -13.4793510, 13.4552383
3: -1.1422013, 11.3195534, -1.1312307, 11.3189182, -12.0549431, 12.0204773
4: -11.1275997, 5.4927402, -11.1157341, 5.4919071, -14.6867828, 14.6846313
5: 1.8237972, 17.7543736, 1.8403587, 17.7546692, -15.9308720, 15.9140148
6: -39.9357529, -18.2170010, -39.9336700, -18.2139721, -15.1293411, 15.2079201
7: -3.6126623, 12.2665272, -3.5822563, 12.2663660, -13.6529388, 13.6267281
8: -6.7255230, 8.5763569, -6.7097721, 8.5766392, -12.1263466, 12.1006126
9: -4.7930636, 11.7246599, -4.7941656, 11.7204628, -13.0280647, 13.0050812
10: 1.2936740, 25.7508583, 1.2946963, 25.7452240, -20.9398270, 20.9585342
11: -11.5111036, 4.2889628, -11.5109482, 4.2885647, -15.7996683, 15.7999115
12: -11.9151468, 9.8594065, -11.9163208, 9.8310013, -15.0183258, 15.0672913
13: -18.5596409, 6.7378492, -18.5673313, 6.7301207, -16.6337776, 16.5797729
14: 4.9228401, 36.4343529, 4.9263420, 36.4221230, -26.7650757, 26.7380295
15: -8.7019072, 9.2939377, -8.7061243, 9.2934160, -17.9953232, 18.0000610
16: -16.7629681, 2.5490603, -16.7460136, 2.5480788, -14.8289185, 14.8238640
17: 6.1849432, 30.6786518, 6.1869254, 30.6595650, -17.2491074, 17.2602119
18: -14.4054718, 5.1338663, -14.3987131, 5.1335173, -14.4214783, 14.4244041
19: -20.2848606, -4.3199253, -20.2822132, -4.3172369, -14.5520020, 14.5513229
20: -2.4302695, 11.2298412, -2.4285638, 11.2287121, -12.6290512, 12.6344948
21: -11.0884476, 3.2661176, -11.0847654, 3.2530284, -14.3414764, 14.3508835
22: -3.7084496, 13.1479950, -3.7071233, 13.1174688, -14.9616547, 14.9872704
23: -14.5892372, 0.3534243, -14.5866003, 0.3536997, -14.3310890, 14.3205147
24: -19.9409313, -5.1101885, -19.9400692, -5.1112013, -9.2716713, 9.2849693
25: -5.4708996, 10.8876858, -5.4677763, 10.8638115, -13.8158226, 13.8353539
26: -21.0270462, 1.2571394, -21.0263977, 1.2161007, -19.3583527, 19.3683624
27: -16.0267868, 2.1873035, -16.0126495, 2.1880269, -13.2239456, 13.2281494
28: -12.8040524, 4.6519713, -12.8015308, 4.6510191, -17.4550705, 17.4535027
29: -5.6019773, 11.9266253, -5.6016922, 11.8933487, -14.9690247, 14.9864006
30: -10.0568380, 6.2237530, -10.0572157, 6.2088842, -13.5586395, 13.5717850
31: -10.9898844, 6.9550982, -10.9818773, 6.9560466, -14.6555328, 14.6547089
32: -24.9306736, -4.5514250, -24.9267273, -4.5509434, -13.2841339, 13.3142586
33: -69.3227081, -40.0843277, -69.3157578, -40.0869713, -16.6406593, 16.6725197
34: -53.7680969, -30.8880119, -53.7648239, -30.8908081, -14.1198311, 14.1636467
35: -47.8115005, -26.0614281, -47.8155518, -26.0559521, -12.9982872, 13.0129356
36: -42.8107948, -19.2713356, -42.8160744, -19.2667942, -15.1123047, 15.1102448
37: -86.6747589, -55.5381699, -86.6751251, -55.5348434, -18.9174919, 18.9408493
38: -52.9551353, -24.3127632, -52.9532776, -24.3134270, -18.3541412, 18.3882599
39: -76.5525284, -44.6165848, -76.5551300, -44.6153412, -16.0862961, 16.0962753
40: -67.2881165, -43.5072250, -67.2546082, -43.5073700, -14.3228836, 14.3507099
41: -55.4443817, -32.9366951, -55.4319611, -32.9386063, -16.6691780, 16.7048798
42: -29.4677238, -9.8826275, -29.4717560, -9.8777161, -17.2402306, 17.2778778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 889

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5694855, upper bound: 12.5536343
time: 6.92 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5704236, upper bound: 12.5704234
time: 33.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 42.27 seconds
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5359199, upper bound: 12.5527998
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5367353, upper bound: 12.5696245
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5693603, upper bound: 12.5372047
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5703000, upper bound: 12.5537152
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5694855, upper bound: 12.5536343
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 42.27
Output dim: 14, lower bound: -12.5704236, upper bound: 12.5704234

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1225567, 3.6726522, -12.1235485, 3.6774135, -13.8722610, 13.8808594
1: -3.6704245, 7.3937216, -3.6709957, 7.3977957, -8.4757118, 8.4951763
2: -0.7431712, 13.4267445, -0.7448494, 13.4339752, -13.4349289, 13.4425888
3: -1.1291590, 11.3061476, -1.1305320, 11.3127422, -12.0259933, 12.0102615
4: -11.1135654, 5.4811640, -11.1146183, 5.4862013, -14.6554947, 14.6731834
5: 1.8434496, 17.7418747, 1.8418722, 17.7488823, -15.9054327, 15.9000025
6: -39.9301682, -18.2216587, -39.9317093, -18.2194824, -15.1419830, 15.1947746
7: -3.5795236, 12.2458572, -3.5806065, 12.2541924, -13.5762939, 13.6106415
8: -6.7053080, 8.5658636, -6.7075806, 8.5710649, -12.1045609, 12.0909882
9: -4.7899752, 11.7158365, -4.7944193, 11.7172537, -13.0183716, 12.9999428
10: 1.3081551, 25.7418213, 1.3006539, 25.7428989, -20.9236069, 20.9521713
11: -11.5037861, 4.2869878, -11.5077953, 4.2876344, -15.7914200, 15.7947826
12: -11.8972645, 9.8283443, -11.9074211, 9.8296270, -15.0007896, 15.0035667
13: -18.5647774, 6.7260227, -18.5671272, 6.7282648, -16.5976105, 16.6444550
14: 4.9530659, 36.4207153, 4.9418859, 36.4214706, -26.7326431, 26.7341995
15: -8.6973686, 9.2893505, -8.6969709, 9.2919779, -17.9893456, 17.9863205
16: -16.7393913, 2.5296762, -16.7434883, 2.5359449, -14.8066559, 14.8069038
17: 6.2100201, 30.6574116, 6.1989198, 30.6582451, -17.2224770, 17.2199783
18: -14.3942070, 5.1240578, -14.3962517, 5.1291351, -14.4102631, 14.4132252
19: -20.2759476, -4.3190498, -20.2797394, -4.3173862, -14.5420761, 14.5400696
20: -2.4232025, 11.2279911, -2.4253650, 11.2288437, -12.6211853, 12.6214714
21: -11.0731525, 3.2521217, -11.0788097, 3.2525465, -14.3256989, 14.3309317
22: -3.6866789, 13.1146927, -3.6957226, 13.1162214, -14.9439468, 14.9157143
23: -14.5802250, 0.3524709, -14.5831823, 0.3533382, -14.3290863, 14.3060036
24: -19.9351006, -5.1129460, -19.9352226, -5.1116633, -9.2639275, 9.2681847
25: -5.4472289, 10.8623219, -5.4526148, 10.8631477, -13.8001862, 13.7791786
26: -21.0006371, 1.2131593, -21.0131340, 1.2149868, -19.3389664, 19.2708817
27: -16.0099678, 2.1790953, -16.0104790, 2.1830542, -13.2359467, 13.2113609
28: -12.7945099, 4.6492186, -12.7981606, 4.6503396, -17.4448490, 17.4473801
29: -5.5820866, 11.8913326, -5.5915036, 11.8924351, -14.9516754, 14.9105759
30: -10.0469532, 6.2074976, -10.0526390, 6.2081432, -13.5488625, 13.5404739
31: -10.9767179, 6.9531927, -10.9793358, 6.9550972, -14.6553841, 14.6449165
32: -24.9242668, -4.5608921, -24.9256096, -4.5605736, -13.2938461, 13.3007622
33: -69.3135071, -40.0951653, -69.3145828, -40.0922241, -16.6204338, 16.6546097
34: -53.7631035, -30.9011326, -53.7640381, -30.8970108, -14.1398964, 14.1440010
35: -47.8196106, -26.0596981, -47.8198853, -26.0585594, -12.9979591, 13.0040398
36: -42.8212395, -19.2729321, -42.8226471, -19.2682571, -15.1094170, 15.0860062
37: -86.6750412, -55.5432892, -86.6776276, -55.5386429, -18.9128265, 18.9183655
38: -52.9479523, -24.3182564, -52.9490280, -24.3156815, -18.3453369, 18.3755493
39: -76.5591965, -44.6190834, -76.5609894, -44.6181297, -16.0800323, 16.0844536
40: -67.2520599, -43.5330734, -67.2531891, -43.5252838, -14.3576775, 14.3278313
41: -55.4300919, -32.9564400, -55.4308281, -32.9523582, -16.7002792, 16.6790161
42: -29.4696541, -9.8737497, -29.4706154, -9.8741159, -17.2720184, 17.2506638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 885

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4979024, upper bound: 12.5677564
time: 26.61 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5359279, upper bound: 12.5688415
time: 7.47 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -12.1166077, 3.6458769, -12.1083374, 3.6467509, -13.8535538, 13.8389435
1: -3.6660657, 7.3801003, -3.6571598, 7.3779874, -8.4719048, 8.4654827
2: -0.7540871, 13.4188194, -0.7267495, 13.4114151, -13.4343033, 13.4101105
3: -1.1266832, 11.2989969, -1.1229349, 11.2963581, -12.0080338, 12.0025177
4: -11.0965223, 5.4534039, -11.0891533, 5.4622998, -14.6358643, 14.6203308
5: 1.8435955, 17.7416229, 1.8548412, 17.7388496, -15.8952541, 15.8867817
6: -39.8590012, -18.2617950, -39.8927307, -18.2188339, -15.1129074, 15.1209831
7: -3.5762601, 12.2309017, -3.5597684, 12.2319965, -13.6006470, 13.5606117
8: -6.7191315, 8.5565176, -6.7004056, 8.5545073, -12.0992584, 12.0906754
9: -4.7588930, 11.6804523, -4.7847352, 11.6868734, -12.9580231, 12.9725571
10: 1.3257732, 25.7323151, 1.3145928, 25.7246342, -20.8838959, 20.8836823
11: -11.4909897, 4.2856803, -11.4978828, 4.2837858, -15.7747755, 15.7835636
12: -11.8760386, 9.8406448, -11.8728971, 9.8175592, -14.9747963, 15.0054207
13: -18.5466976, 6.7065706, -18.5518990, 6.7174764, -16.6345139, 16.5280457
14: 4.9806061, 36.3969460, 4.9730253, 36.4055290, -26.7034531, 26.6827774
15: -8.6627302, 9.2253504, -8.6989622, 9.2727289, -17.9354591, 17.9243126
16: -16.7343445, 2.5247416, -16.7310104, 2.5064387, -14.7459831, 14.8016968
17: 6.2439480, 30.6516132, 6.2467356, 30.6348934, -17.1661720, 17.1730156
18: -14.3640881, 5.1122017, -14.3702297, 5.1187549, -14.3649902, 14.3710995
19: -20.2580643, -4.3325629, -20.2602673, -4.3272653, -14.5075111, 14.5147629
20: -2.3994026, 11.2166367, -2.3990214, 11.2191143, -12.5969772, 12.5965843
21: -11.0578556, 3.2595301, -11.0561352, 3.2441554, -14.3020115, 14.3156652
22: -3.6679535, 13.1057873, -3.6635599, 13.0812416, -14.8742104, 14.9091339
23: -14.5497036, 0.3100317, -14.5680161, 0.3284066, -14.2611923, 14.2804184
24: -19.9329739, -5.1208930, -19.9324627, -5.1111207, -9.2568169, 9.2532387
25: -5.4425097, 10.8586750, -5.4377022, 10.8504486, -13.7579575, 13.7759399
26: -20.9683800, 1.2023528, -20.9667587, 1.1612167, -19.2517281, 19.2920074
27: -16.0074177, 2.1725302, -16.0040245, 2.1764069, -13.1734581, 13.2098846
28: -12.7636585, 4.6093464, -12.7807512, 4.6224513, -17.3861103, 17.3900986
29: -5.5378466, 11.8776550, -5.5507202, 11.8457680, -14.8570251, 14.9062729
30: -10.0363503, 6.2155113, -10.0343342, 6.1943936, -13.5172615, 13.5378342
31: -10.9556589, 6.9532695, -10.9606171, 6.9551058, -14.6090584, 14.6305542
32: -24.8950214, -4.5718842, -24.9162445, -4.5578136, -13.2735214, 13.2876549
33: -69.2883453, -40.1324768, -69.2858353, -40.1185608, -16.6031456, 16.5966034
34: -53.7331200, -30.9198265, -53.7421341, -30.9110432, -14.1112213, 14.1308556
35: -47.8002472, -26.0744781, -47.8048630, -26.0604572, -12.9865494, 12.9829483
36: -42.7971497, -19.2780876, -42.7877426, -19.2761631, -15.0768280, 15.0715485
37: -86.6620026, -55.5596123, -86.6609573, -55.5570869, -18.8868561, 18.8958092
38: -52.8992081, -24.3465309, -52.9102631, -24.3159218, -18.3239861, 18.3068542
39: -76.5230560, -44.6489143, -76.5345001, -44.6366081, -16.0344391, 16.0453262
40: -67.2467422, -43.5356064, -67.2287827, -43.5424843, -14.2714348, 14.3359699
41: -55.4199829, -32.9527054, -55.4263229, -32.9553642, -16.6487694, 16.6987648
42: -29.4526978, -9.8871880, -29.4666138, -9.8824062, -17.2061806, 17.2662621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673755, upper bound: 12.4994731
time: 9.40 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5685761, upper bound: 12.5364181
time: 15.95 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -12.1184177, 3.6587777, -12.1188259, 3.6679106, -13.8617935, 13.8640976
1: -3.6663570, 7.3922806, -3.6613965, 7.3979120, -8.4770355, 8.4855518
2: -0.7553254, 13.4329815, -0.7391909, 13.4352455, -13.4472618, 13.4365845
3: -1.1274658, 11.3089809, -1.1225990, 11.3133297, -12.0219574, 12.0118160
4: -11.0972147, 5.4672298, -11.0979214, 5.4856834, -14.6481781, 14.6428947
5: 1.8423038, 17.7489548, 1.8511944, 17.7514229, -15.9091187, 15.8977604
6: -39.8607750, -18.2633533, -39.8894310, -18.2222214, -15.1319580, 15.1165123
7: -3.5769272, 12.2470455, -3.5622902, 12.2589140, -13.6105347, 13.5957413
8: -6.7198515, 8.5658426, -6.7073636, 8.5698156, -12.1169586, 12.0864105
9: -4.7601027, 11.6869125, -4.7872505, 11.6982832, -12.9739227, 12.9873161
10: 1.3237133, 25.7402420, 1.3112531, 25.7391758, -20.9002914, 20.9126587
11: -11.4942398, 4.2863970, -11.5028543, 4.2863407, -15.7805805, 15.7892513
12: -11.8949928, 9.8419628, -11.9046078, 9.8285151, -15.0048447, 15.0213203
13: -18.5507393, 6.7076850, -18.5581207, 6.7138052, -16.6106720, 16.5820808
14: 4.9598150, 36.3974457, 4.9372168, 36.4013443, -26.7050095, 26.7312393
15: -8.6589470, 9.2262068, -8.6911135, 9.2543936, -17.9133415, 17.9173203
16: -16.7356892, 2.5465493, -16.7324200, 2.5423427, -14.7809181, 14.8091812
17: 6.2117949, 30.6522293, 6.1923060, 30.6450539, -17.2064247, 17.2171288
18: -14.3674583, 5.1171198, -14.3763294, 5.1273184, -14.3754158, 14.3828583
19: -20.2654152, -4.3323908, -20.2731476, -4.3246875, -14.5239716, 14.5287704
20: -2.4098644, 11.2178307, -2.4165692, 11.2255001, -12.6130791, 12.6083908
21: -11.0673027, 3.2606153, -11.0724697, 3.2518976, -14.3192005, 14.3330851
22: -3.6904037, 13.1062946, -3.7004764, 13.0932989, -14.9157410, 14.9254227
23: -14.5566902, 0.3103461, -14.5807533, 0.3283420, -14.2776871, 14.2928352
24: -19.9330521, -5.1202478, -19.9332428, -5.1165242, -9.2675591, 9.2599945
25: -5.4539723, 10.8591690, -5.4575863, 10.8484936, -13.7860603, 13.7894592
26: -21.0003967, 1.2033358, -21.0205421, 1.1849787, -19.3092880, 19.3046646
27: -16.0086823, 2.1741881, -16.0054131, 2.1792631, -13.2001381, 13.2067566
28: -12.7721424, 4.6097193, -12.7962952, 4.6261697, -17.3983116, 17.4060135
29: -5.5651073, 11.8779850, -5.5964370, 11.8651018, -14.9048233, 14.9224052
30: -10.0467901, 6.2167578, -10.0515604, 6.2037539, -13.5371895, 13.5459137
31: -10.9576035, 6.9535027, -10.9646168, 6.9551091, -14.6298485, 14.6341057
32: -24.8962193, -4.5739059, -24.9067898, -4.5609241, -13.2942085, 13.2760696
33: -69.2889709, -40.1190414, -69.2957535, -40.0953751, -16.6112518, 16.6124763
34: -53.7339439, -30.9111710, -53.7441711, -30.8958282, -14.1438065, 14.1152077
35: -47.8013954, -26.0731926, -47.8078842, -26.0600662, -13.0018158, 12.9925003
36: -42.8100815, -19.2771988, -42.8103523, -19.2691536, -15.1013031, 15.0772972
37: -86.6659775, -55.5511627, -86.6690674, -55.5431900, -18.9175110, 18.9091835
38: -52.9076233, -24.3457451, -52.9247627, -24.3164177, -18.3381271, 18.3240433
39: -76.5238266, -44.6396599, -76.5392456, -44.6210556, -16.0527458, 16.0513687
40: -67.2480621, -43.5189438, -67.2342529, -43.5155029, -14.3059654, 14.3053093
41: -55.4206009, -32.9480438, -55.4193535, -32.9473343, -16.6814575, 16.6786995
42: -29.4533825, -9.8878098, -29.4640961, -9.8836470, -17.2384148, 17.2598038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5684757, upper bound: 12.5166702
time: 20.58 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5695172, upper bound: 12.5529336
time: 6.73 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -12.1396446, 3.6686530, -12.1136713, 3.6598830, -13.8901482, 13.8639030
1: -3.6827507, 7.3903480, -3.6669979, 7.3812914, -8.4822578, 8.4815350
2: -0.7672205, 13.4255314, -0.7332075, 13.4153986, -13.4528427, 13.4275551
3: -1.1412108, 11.3092346, -1.1311790, 11.3012924, -12.0359612, 12.0101509
4: -11.1265984, 5.4778929, -11.1063385, 5.4666533, -14.6605988, 14.6603813
5: 1.8256474, 17.7465820, 1.8449326, 17.7412224, -15.9155750, 15.9016495
6: -39.9330673, -18.2198372, -39.9353981, -18.2184658, -15.1176376, 15.2062111
7: -3.6113493, 12.2476196, -3.5785182, 12.2345438, -13.6148376, 13.5893936
8: -6.7240248, 8.5660095, -6.7013826, 8.5595980, -12.1117096, 12.1008568
9: -4.7914953, 11.7170496, -4.7910175, 11.7069187, -13.0097046, 12.9898872
10: 1.2963104, 25.7422924, 1.2991037, 25.7294998, -20.9188156, 20.9346466
11: -11.5072174, 4.2879128, -11.5048447, 4.2854004, -15.7926178, 15.7927570
12: -11.8952799, 9.8577909, -11.8829584, 9.8195457, -14.9867287, 15.0331192
13: -18.5511913, 6.7358246, -18.5532112, 6.7321167, -16.6451912, 16.5571747
14: 4.9477425, 36.4335899, 4.9694023, 36.4258156, -26.7570648, 26.6968842
15: -8.6988096, 9.2926979, -8.7015486, 9.3110638, -18.0098724, 17.9942474
16: -16.7611008, 2.5247216, -16.7437000, 2.5076365, -14.7825966, 14.8140106
17: 6.2189469, 30.6775990, 6.2446504, 30.6486645, -17.2062073, 17.1999245
18: -14.4013815, 5.1283045, -14.3913078, 5.1238217, -14.4066010, 14.4105492
19: -20.2769527, -4.3202205, -20.2683716, -4.3200040, -14.5363731, 14.5352478
20: -2.4188914, 11.2284985, -2.4094348, 11.2220898, -12.6114807, 12.6148376
21: -11.0781937, 3.2648895, -11.0671282, 3.2450385, -14.3232327, 14.3320179
22: -3.6839037, 13.1470213, -3.6665511, 13.1045475, -14.9179001, 14.9429016
23: -14.5817289, 0.3528957, -14.5729408, 0.3533900, -14.3163757, 14.3055954
24: -19.9379883, -5.1109734, -19.9342194, -5.1060843, -9.2597656, 9.2711563
25: -5.4553585, 10.8868370, -5.4405708, 10.8651657, -13.7855682, 13.7991714
26: -20.9934826, 1.2559557, -20.9698982, 1.1919847, -19.2988739, 19.3106232
27: -16.0246849, 2.1843138, -16.0097351, 2.1828384, -13.2083473, 13.2254028
28: -12.7951298, 4.6513577, -12.7852364, 4.6469202, -17.4420509, 17.4365940
29: -5.5731077, 11.9259758, -5.5530872, 11.8734131, -14.9189987, 14.9373970
30: -10.0455112, 6.2221885, -10.0383997, 6.1989927, -13.5372009, 13.5515289
31: -10.9873981, 6.9547253, -10.9768000, 6.9558010, -14.6412926, 14.6462212
32: -24.9290562, -4.5545654, -24.9354305, -4.5571532, -13.2744637, 13.3195534
33: -69.3216705, -40.0992012, -69.3050613, -40.1128540, -16.6151047, 16.6476784
34: -53.7670364, -30.8977299, -53.7624283, -30.9079399, -14.1075745, 14.1719208
35: -47.8086319, -26.0636158, -47.8094749, -26.0579910, -12.9845352, 12.9997826
36: -42.7968445, -19.2726574, -42.7917328, -19.2746220, -15.0858803, 15.0823555
37: -86.6702728, -55.5479584, -86.6661377, -55.5510406, -18.8937759, 18.9232750
38: -52.9450607, -24.3138390, -52.9357071, -24.3135223, -18.3381882, 18.3651886
39: -76.5514832, -44.6270599, -76.5498505, -44.6330948, -16.0665550, 16.0877228
40: -67.2861328, -43.5280991, -67.2479858, -43.5420570, -14.3098793, 14.3700142
41: -55.4434052, -32.9456711, -55.4382439, -32.9544258, -16.6577148, 16.7170830
42: -29.4665203, -9.8847094, -29.4733849, -9.8813524, -17.2231674, 17.2775421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675031, upper bound: 12.5153521
time: 10.36 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5687014, upper bound: 12.5528415
time: 22.72 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -12.1414433, 3.6815472, -12.1241570, 3.6810431, -13.8984108, 13.8890610
1: -3.6830468, 7.4025459, -3.6712413, 7.4012117, -8.4874115, 8.5015965
2: -0.7684686, 13.4396801, -0.7456371, 13.4392538, -13.4657364, 13.4540100
3: -1.1419917, 11.3191833, -1.1308613, 11.3182459, -12.0498314, 12.0194817
4: -11.1272745, 5.4917088, -11.1151047, 5.4900131, -14.6729507, 14.6829414
5: 1.8243141, 17.7538834, 1.8412819, 17.7538033, -15.9294891, 15.9126015
6: -39.9348450, -18.2214050, -39.9320755, -18.2218685, -15.1366577, 15.2017136
7: -3.6119983, 12.2637625, -3.5810592, 12.2614632, -13.6247177, 13.6245384
8: -6.7247334, 8.5753689, -6.7083263, 8.5749016, -12.1294022, 12.0965805
9: -4.7926922, 11.7234831, -4.7935138, 11.7183485, -13.0256157, 13.0046539
10: 1.2942352, 25.7501926, 1.2957497, 25.7440529, -20.9352646, 20.9635773
11: -11.5104847, 4.2886162, -11.5098209, 4.2879596, -15.7984447, 15.7984371
12: -11.9142103, 9.8591347, -11.9146557, 9.8304796, -15.0167770, 15.0490303
13: -18.5552826, 6.7369633, -18.5594521, 6.7284575, -16.6213608, 16.6112213
14: 4.9268970, 36.4340706, 4.9335575, 36.4216309, -26.7586365, 26.7453766
15: -8.6950550, 9.2935524, -8.6936874, 9.2927380, -17.9877930, 17.9872398
16: -16.7624321, 2.5465446, -16.7451019, 2.5435233, -14.8175659, 14.8214188
17: 6.1867609, 30.6782131, 6.1901989, 30.6588020, -17.2464790, 17.2440643
18: -14.4047279, 5.1332293, -14.3973999, 5.1323528, -14.4170036, 14.4222946
19: -20.2842979, -4.3200188, -20.2812424, -4.3174124, -14.5528183, 14.5492210
20: -2.4294136, 11.2296925, -2.4269981, 11.2284679, -12.6275787, 12.6266556
21: -11.0876350, 3.2659831, -11.0834799, 3.2528062, -14.3404408, 14.3494625
22: -3.7063673, 13.1475306, -3.7034519, 13.1165972, -14.9594231, 14.9591637
23: -14.5887089, 0.3531914, -14.5856848, 0.3533344, -14.3328857, 14.3180122
24: -19.9380531, -5.1103354, -19.9349937, -5.1114788, -9.2704926, 9.2779007
25: -5.4668446, 10.8873444, -5.4604588, 10.8632002, -13.8136673, 13.8126717
26: -21.0255203, 1.2569172, -21.0236511, 1.2157083, -19.3564110, 19.3232803
27: -16.0259361, 2.1859789, -16.0111351, 2.1856861, -13.2349968, 13.2222672
28: -12.8036070, 4.6517415, -12.8007545, 4.6506352, -17.4542427, 17.4524956
29: -5.6003814, 11.9263306, -5.5988188, 11.8927469, -14.9668350, 14.9535294
30: -10.0559368, 6.2234249, -10.0556574, 6.2083831, -13.5571480, 13.5596199
31: -10.9893379, 6.9549508, -10.9808426, 6.9557881, -14.6620903, 14.6497536
32: -24.9302540, -4.5565815, -24.9259300, -4.5602961, -13.2951508, 13.3079987
33: -69.3223267, -40.0857811, -69.3150482, -40.0896225, -16.6231613, 16.6635895
34: -53.7679214, -30.8890915, -53.7644272, -30.8927727, -14.1401596, 14.1562653
35: -47.8098335, -26.0623169, -47.8124924, -26.0576134, -12.9997940, 13.0093651
36: -42.8098221, -19.2718067, -42.8143311, -19.2675838, -15.1103706, 15.0881424
37: -86.6742859, -55.5395203, -86.6743011, -55.5371780, -18.9244614, 18.9366074
38: -52.9533882, -24.3130665, -52.9501801, -24.3140144, -18.3523369, 18.3823853
39: -76.5522003, -44.6178055, -76.5545197, -44.6175079, -16.0848961, 16.0937309
40: -67.2874451, -43.5114861, -67.2534256, -43.5150719, -14.3443909, 14.3393764
41: -55.4440346, -32.9409866, -55.4312935, -32.9463882, -16.6903572, 16.6970367
42: -29.4671955, -9.8853283, -29.4708080, -9.8826056, -17.2553711, 17.2710800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5686007, upper bound: 12.5330612
time: 7.67 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5696408, upper bound: 12.5696404
time: 17.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.99 seconds
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.4979024, upper bound: 12.5677564
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5359279, upper bound: 12.5688415
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5673755, upper bound: 12.4994731
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5685761, upper bound: 12.5364181
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5684757, upper bound: 12.5166702
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5695172, upper bound: 12.5529336
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5675031, upper bound: 12.5153521
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5687014, upper bound: 12.5528415
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5686007, upper bound: 12.5330612
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 14, lower bound: -12.5696408, upper bound: 12.5696404

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.1189508, 3.6728430, -12.1213512, 3.6766407, -13.8678818, 13.8786697
1: -3.6632962, 7.3909497, -3.6658487, 7.3965540, -8.4675827, 8.4877834
2: -0.7420887, 13.4228010, -0.7440266, 13.4317245, -13.4314728, 13.4363823
3: -1.1268005, 11.2961693, -1.1297023, 11.3068867, -12.0168304, 11.9994698
4: -11.1095123, 5.4791965, -11.1122551, 5.4840531, -14.6466675, 14.6636963
5: 1.8425927, 17.7345009, 1.8426533, 17.7444077, -15.9018154, 15.8918476
6: -39.9163551, -18.2431755, -39.9297943, -18.2322617, -15.1146088, 15.1712265
7: -3.5769508, 12.2416725, -3.5793290, 12.2518024, -13.5677032, 13.6049576
8: -6.7019057, 8.5635700, -6.7051301, 8.5698366, -12.0989685, 12.0807686
9: -4.7731099, 11.7122478, -4.7838521, 11.7165995, -13.0008469, 12.9878044
10: 1.3386350, 25.7309208, 1.3185143, 25.7417870, -20.8914261, 20.9230347
11: -11.4955702, 4.2870331, -11.5025177, 4.2874756, -15.7830458, 15.7895508
12: -11.8837423, 9.8233376, -11.8992300, 9.8286743, -14.9856911, 14.9890366
13: -18.5503483, 6.7200036, -18.5579777, 6.7260146, -16.5758476, 16.6202660
14: 4.9924612, 36.4049492, 4.9648199, 36.4204102, -26.6913223, 26.6920853
15: -8.6880512, 9.2826967, -8.6915874, 9.2890606, -17.9771118, 17.9742851
16: -16.7189960, 2.5318153, -16.7294216, 2.5352449, -14.7875023, 14.8031845
17: 6.2299557, 30.6484337, 6.2102475, 30.6573391, -17.2027931, 17.2103195
18: -14.3927174, 5.1188483, -14.3947067, 5.1259928, -14.4059563, 14.4058151
19: -20.2705383, -4.3251119, -20.2767487, -4.3216114, -14.5311356, 14.5303917
20: -2.4155664, 11.2134666, -2.4233215, 11.2194958, -12.6071510, 12.6057396
21: -11.0679579, 3.2496500, -11.0755320, 3.2505910, -14.3185492, 14.3251820
22: -3.6848326, 13.1084194, -3.6945109, 13.1109753, -14.9351540, 14.9067383
23: -14.5770721, 0.3444924, -14.5803356, 0.3479505, -14.3185272, 14.2942619
24: -19.9316425, -5.1138783, -19.9331932, -5.1124668, -9.2593880, 9.2648201
25: -5.4393520, 10.8593826, -5.4480782, 10.8624115, -13.7908401, 13.7728996
26: -20.9943619, 1.2093093, -21.0095596, 1.2129555, -19.3284912, 19.2623672
27: -15.9995184, 2.1629016, -16.0090599, 2.1731801, -13.2139549, 13.1941261
28: -12.7868614, 4.6370258, -12.7959156, 4.6430578, -17.4299202, 17.4329414
29: -5.5792217, 11.8882275, -5.5900106, 11.8902683, -14.9420204, 14.9051514
30: -10.0386848, 6.2051954, -10.0474348, 6.2074509, -13.5408516, 13.5361481
31: -10.9710693, 6.9473314, -10.9758854, 6.9506040, -14.6462326, 14.6354637
32: -24.9086151, -4.5839386, -24.9240646, -4.5745158, -13.2658386, 13.2774162
33: -69.3055115, -40.1119308, -69.3131409, -40.1024323, -16.6145554, 16.6431541
34: -53.7518845, -30.9231434, -53.7630348, -30.9100647, -14.1144218, 14.1194572
35: -47.8149719, -26.0698433, -47.8191605, -26.0652866, -12.9864922, 12.9934845
36: -42.8132439, -19.2925091, -42.8217468, -19.2802715, -15.0898170, 15.0664406
37: -86.6704254, -55.5511246, -86.6759338, -55.5434074, -18.9073868, 18.9087906
38: -52.9331627, -24.3437347, -52.9480171, -24.3312950, -18.3138351, 18.3494186
39: -76.5558853, -44.6250648, -76.5591278, -44.6218948, -16.0732269, 16.0750961
40: -67.2428436, -43.5478745, -67.2513962, -43.5341339, -14.3327942, 14.3099937
41: -55.4164810, -32.9782639, -55.4292297, -32.9653130, -16.6719322, 16.6537056
42: -29.4575348, -9.8924179, -29.4688225, -9.8851614, -17.2500458, 17.2300301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4567553, upper bound: 12.5673752
time: 13.01 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4975104, upper bound: 12.5673752
time: 7.60 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1213398, 3.6722493, -12.1228447, 3.6771932, -13.8711281, 13.8791695
1: -3.6685519, 7.3933783, -3.6698849, 7.3975940, -8.4723282, 8.4936600
2: -0.7429612, 13.4253273, -0.7447245, 13.4331722, -13.4335365, 13.4430733
3: -1.1289231, 11.3049545, -1.1304035, 11.3120613, -12.0250320, 12.0062943
4: -11.1114864, 5.4802208, -11.1134396, 5.4856715, -14.6539612, 14.6732597
5: 1.8436632, 17.7403107, 1.8420072, 17.7479782, -15.9043150, 15.8983040
6: -39.9282761, -18.2221470, -39.9306221, -18.2197533, -15.1390152, 15.1666069
7: -3.5789862, 12.2453470, -3.5802815, 12.2538891, -13.5729599, 13.6031227
8: -6.7044420, 8.5652800, -6.7070570, 8.5707273, -12.1017685, 12.0936718
9: -4.7868958, 11.7155008, -4.7926817, 11.7170668, -13.0071602, 12.9983749
10: 1.3113337, 25.7411633, 1.3024716, 25.7425385, -20.9078827, 20.9495850
11: -11.5022850, 4.2868805, -11.5069427, 4.2875867, -15.7898712, 15.7938232
12: -11.8954105, 9.8279705, -11.9063683, 9.8294144, -14.9911842, 15.0021591
13: -18.5604210, 6.7252355, -18.5646973, 6.7277851, -16.5715714, 16.6435661
14: 4.9567032, 36.4201584, 4.9439363, 36.4211617, -26.7063904, 26.7313080
15: -8.6969652, 9.2883358, -8.6967258, 9.2913895, -17.9883537, 17.9850616
16: -16.7361488, 2.5294216, -16.7415886, 2.5358059, -14.8044815, 14.8036499
17: 6.2121997, 30.6569195, 6.2001724, 30.6579361, -17.2137871, 17.2179298
18: -14.3937798, 5.1225657, -14.3960037, 5.1282682, -14.4081001, 14.4146538
19: -20.2752304, -4.3204370, -20.2793083, -4.3182039, -14.5406647, 14.5377884
20: -2.4226687, 11.2270603, -2.4250534, 11.2282715, -12.6200409, 12.6145630
21: -11.0723248, 3.2507625, -11.0783062, 3.2517810, -14.3241062, 14.3290691
22: -3.6863201, 13.1129789, -3.6955194, 13.1152344, -14.9426079, 14.9129066
23: -14.5795574, 0.3498428, -14.5827894, 0.3517737, -14.3275986, 14.3027496
24: -19.9339085, -5.1131935, -19.9345551, -5.1118102, -9.2628632, 9.2675629
25: -5.4463234, 10.8619986, -5.4521089, 10.8629742, -13.7958832, 13.7759933
26: -20.9996490, 1.2123830, -21.0125465, 1.2145255, -19.3359604, 19.2681808
27: -16.0094719, 2.1774647, -16.0101948, 2.1821306, -13.2343826, 13.1877518
28: -12.7936831, 4.6477213, -12.7976627, 4.6494684, -17.4431515, 17.4453850
29: -5.5813437, 11.8900871, -5.5910978, 11.8917227, -14.9531059, 14.9079475
30: -10.0449753, 6.2072449, -10.0514965, 6.2079954, -13.5461197, 13.5385857
31: -10.9756088, 6.9507971, -10.9787083, 6.9537377, -14.6530190, 14.6413803
32: -24.9234791, -4.5616455, -24.9251232, -4.5610132, -13.2921982, 13.2781868
33: -69.3127670, -40.0969086, -69.3141479, -40.0931931, -16.6163559, 16.6499252
34: -53.7622643, -30.9030190, -53.7635384, -30.8982296, -14.1379356, 14.1242409
35: -47.8190918, -26.0609684, -47.8195992, -26.0593224, -12.9972267, 13.0021515
36: -42.8200989, -19.2753525, -42.8219337, -19.2697353, -15.1068840, 15.0737877
37: -86.6739655, -55.5439453, -86.6770020, -55.5390396, -18.9102325, 18.9190369
38: -52.9470596, -24.3200474, -52.9485130, -24.3167171, -18.3434334, 18.3499069
39: -76.5580902, -44.6197472, -76.5603333, -44.6185532, -16.0775223, 16.0860863
40: -67.2504578, -43.5339966, -67.2522278, -43.5257950, -14.3517113, 14.3020821
41: -55.4289474, -32.9575500, -55.4301605, -32.9530449, -16.6985168, 16.6529922
42: -29.4682541, -9.8742676, -29.4697971, -9.8744202, -17.2703857, 17.2302513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4946436, upper bound: 12.5684410
time: 7.56 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5355290, upper bound: 12.5684410
time: 6.33 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -12.1144400, 3.6451509, -12.1047115, 3.6469517, -13.8513718, 13.8345413
1: -3.6609182, 7.3788385, -3.6500282, 7.3752060, -8.4644928, 8.4573421
2: -0.7532676, 13.4165506, -0.7256662, 13.4074688, -13.4281006, 13.4066658
3: -1.1258581, 11.2931671, -1.1205932, 11.2863922, -11.9972649, 11.9933510
4: -11.0941343, 5.4512539, -11.0851107, 5.4603367, -14.6263580, 14.6115265
5: 1.8443661, 17.7371712, 1.8539877, 17.7314739, -15.8871078, 15.8831835
6: -39.8571167, -18.2745876, -39.8789444, -18.2403469, -15.0893707, 15.0936623
7: -3.5749986, 12.2285423, -3.5571942, 12.2278042, -13.5949707, 13.5519981
8: -6.7166977, 8.5552559, -6.6970110, 8.5522146, -12.0890503, 12.0850945
9: -4.7482929, 11.6797953, -4.7678499, 11.6832962, -12.9459267, 12.9550896
10: 1.3436522, 25.7311592, 1.3450422, 25.7137165, -20.8547592, 20.8515015
11: -11.4856949, 4.2855291, -11.4896545, 4.2838655, -15.7695599, 15.7751837
12: -11.8678350, 9.8396864, -11.8593445, 9.8125706, -14.9602547, 14.9903374
13: -18.5375347, 6.7043138, -18.5374947, 6.7114754, -16.6103058, 16.5063019
14: 5.0035448, 36.3958359, 5.0124311, 36.3897858, -26.6613922, 26.6414719
15: -8.6573582, 9.2224760, -8.6896486, 9.2660723, -17.9234314, 17.9121246
16: -16.7202816, 2.5240574, -16.7106400, 2.5086174, -14.7423515, 14.7825813
17: 6.2553110, 30.6507053, 6.2666721, 30.6258774, -17.1565018, 17.1533432
18: -14.3625450, 5.1090722, -14.3687572, 5.1135483, -14.3575935, 14.3668098
19: -20.2550259, -4.3367734, -20.2548542, -4.3333416, -14.4978371, 14.5037880
20: -2.3973577, 11.2072735, -2.3914170, 11.2045794, -12.5812492, 12.5825272
21: -11.0545664, 3.2575648, -11.0509109, 3.2416778, -14.2962437, 14.3084755
22: -3.6667268, 13.1005449, -3.6617370, 13.0749836, -14.8652687, 14.9004364
23: -14.5468540, 0.3046770, -14.5648108, 0.3204293, -14.2494469, 14.2698746
24: -19.9309578, -5.1216698, -19.9290028, -5.1120276, -9.2534256, 9.2487183
25: -5.4379454, 10.8579226, -5.4298677, 10.8474846, -13.7516403, 13.7665977
26: -20.9648514, 1.2003264, -20.9604816, 1.1573811, -19.2431793, 19.2815323
27: -16.0059910, 2.1626449, -15.9935551, 2.1602168, -13.1562347, 13.1878929
28: -12.7614498, 4.6020246, -12.7730980, 4.6102686, -17.3717194, 17.3751221
29: -5.5363235, 11.8754721, -5.5478811, 11.8426867, -14.8515854, 14.8966064
30: -10.0311718, 6.2147908, -10.0260906, 6.1920872, -13.5129280, 13.5298233
31: -10.9522114, 6.9487915, -10.9549513, 6.9492369, -14.5996017, 14.6213875
32: -24.8935299, -4.5857992, -24.9005642, -4.5808439, -13.2501945, 13.2596512
33: -69.2869263, -40.1427383, -69.2778168, -40.1353760, -16.5916862, 16.5907097
34: -53.7321434, -30.9329281, -53.7309418, -30.9330139, -14.0866852, 14.1053810
35: -47.7994919, -26.0811844, -47.8002548, -26.0705547, -12.9759941, 12.9714775
36: -42.7963028, -19.2900658, -42.7797241, -19.2957115, -15.0572243, 15.0519485
37: -86.6603241, -55.5643616, -86.6563110, -55.5649376, -18.8772888, 18.8903618
38: -52.8981857, -24.3621044, -52.8955612, -24.3414459, -18.2978516, 18.2753601
39: -76.5212097, -44.6527176, -76.5312729, -44.6426315, -16.0250511, 16.0385361
40: -67.2449265, -43.5444489, -67.2195358, -43.5572929, -14.2535858, 14.3110886
41: -55.4183807, -32.9657288, -55.4126854, -32.9771690, -16.6234512, 16.6703873
42: -29.4509392, -9.8982439, -29.4545078, -9.9011002, -17.1856155, 17.2442818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5252697, upper bound: 12.4991007
time: 13.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5670018, upper bound: 12.4991006
time: 31.87 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -12.1159210, 3.6456490, -12.1071339, 3.6463614, -13.8518600, 13.8378220
1: -3.6649396, 7.3799076, -3.6553032, 7.3776264, -8.4703789, 8.4620686
2: -0.7539703, 13.4179974, -0.7265375, 13.4099760, -13.4347992, 13.4087296
3: -1.1265458, 11.2983379, -1.1227150, 11.2951775, -12.0040703, 12.0015697
4: -11.0953236, 5.4528732, -11.0870972, 5.4613380, -14.6359177, 14.6188049
5: 1.8437328, 17.7407379, 1.8550544, 17.7372856, -15.8935528, 15.8856831
6: -39.8579102, -18.2620735, -39.8908501, -18.2193222, -15.0847321, 15.1180115
7: -3.5759492, 12.2306280, -3.5592194, 12.2314768, -13.5930710, 13.5572395
8: -6.7186089, 8.5561876, -6.6995621, 8.5539408, -12.1019630, 12.0878716
9: -4.7571268, 11.6802273, -4.7816715, 11.6865072, -12.9564743, 12.9613800
10: 1.3275785, 25.7319107, 1.3177481, 25.7239647, -20.8812790, 20.8679886
11: -11.4901018, 4.2856288, -11.4963703, 4.2837029, -15.7738047, 15.7819996
12: -11.8749847, 9.8404446, -11.8710117, 9.8171997, -14.9733696, 14.9958191
13: -18.5441914, 6.7060785, -18.5475082, 6.7166777, -16.6336479, 16.5019455
14: 4.9826937, 36.3966370, 4.9767008, 36.4049835, -26.7005615, 26.6565323
15: -8.6624832, 9.2247410, -8.6985626, 9.2716961, -17.9341793, 17.9233036
16: -16.7324505, 2.5246043, -16.7277317, 2.5062222, -14.7427139, 14.7995148
17: 6.2451997, 30.6513405, 6.2489219, 30.6343784, -17.1641235, 17.1643486
18: -14.3638506, 5.1113462, -14.3698158, 5.1172724, -14.3664398, 14.3689518
19: -20.2576408, -4.3333735, -20.2595558, -4.3286734, -14.5052490, 14.5133247
20: -2.3990998, 11.2160530, -2.3985131, 11.2181768, -12.5900688, 12.5954247
21: -11.0573702, 3.2587705, -11.0552950, 3.2428126, -14.3001823, 14.3140659
22: -3.6677451, 13.1047649, -3.6632071, 13.0795412, -14.8714523, 14.9078140
23: -14.5493021, 0.3084812, -14.5673237, 0.3257904, -14.2579803, 14.2789307
24: -19.9323044, -5.1210265, -19.9312706, -5.1113763, -9.2561913, 9.2521858
25: -5.4419889, 10.8584766, -5.4368367, 10.8501244, -13.7547951, 13.7716293
26: -20.9677544, 1.2018845, -20.9657764, 1.1604438, -19.2490082, 19.2890167
27: -16.0071335, 2.1716046, -16.0035248, 2.1747761, -13.1498795, 13.2083244
28: -12.7631884, 4.6084418, -12.7799625, 4.6209426, -17.3841305, 17.3884048
29: -5.5374174, 11.8769379, -5.5499654, 11.8445492, -14.8543854, 14.9076767
30: -10.0352154, 6.2153769, -10.0323658, 6.1941528, -13.5153732, 13.5351181
31: -10.9550238, 6.9519029, -10.9595051, 6.9526887, -14.6055336, 14.6281853
32: -24.8945656, -4.5723057, -24.9154720, -4.5585241, -13.2509422, 13.2860031
33: -69.2878876, -40.1334686, -69.2850647, -40.1203537, -16.5984612, 16.5924873
34: -53.7326355, -30.9210739, -53.7413177, -30.9129372, -14.0914574, 14.1289330
35: -47.7999344, -26.0752487, -47.8042831, -26.0617123, -12.9846458, 12.9822044
36: -42.7964935, -19.2794952, -42.7865829, -19.2785645, -15.0645981, 15.0690002
37: -86.6613617, -55.5600433, -86.6598587, -55.5577774, -18.8875275, 18.8931808
38: -52.8987274, -24.3475876, -52.9094048, -24.3177166, -18.2983284, 18.3049240
39: -76.5223923, -44.6492996, -76.5334320, -44.6372910, -16.0360718, 16.0428429
40: -67.2457962, -43.5361099, -67.2271805, -43.5434113, -14.2456360, 14.3299980
41: -55.4192886, -32.9534531, -55.4251175, -32.9564323, -16.6227226, 16.6970024
42: -29.4519005, -9.8874722, -29.4652309, -9.8829079, -17.1857758, 17.2646103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681810, upper bound: 12.4943138
time: 7.53 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681810, upper bound: 12.5360208
time: 6.87 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -12.1162252, 3.6580231, -12.1151991, 3.6680899, -13.8596001, 13.8596992
1: -3.6612082, 7.3910556, -3.6542683, 7.3951406, -8.4696198, 8.4774227
2: -0.7545109, 13.4306993, -0.7380952, 13.4313450, -13.4410400, 13.4331436
3: -1.1266340, 11.3030996, -1.1202465, 11.3033533, -12.0111961, 12.0026665
4: -11.0948238, 5.4650574, -11.0938587, 5.4837041, -14.6386719, 14.6340637
5: 1.8430572, 17.7445107, 1.8503580, 17.7440739, -15.9010162, 15.8941526
6: -39.8588982, -18.2761478, -39.8756332, -18.2437382, -15.1084137, 15.0891914
7: -3.5756412, 12.2446623, -3.5597100, 12.2547512, -13.6048508, 13.5871506
8: -6.7174101, 8.5645809, -6.7039795, 8.5675125, -12.1067543, 12.0807953
9: -4.7494926, 11.6862497, -4.7703886, 11.6947298, -12.9618263, 12.9698257
10: 1.3416023, 25.7391224, 1.3417039, 25.7282448, -20.8711472, 20.8804550
11: -11.4889374, 4.2862368, -11.4946327, 4.2864146, -15.7753525, 15.7808695
12: -11.8867893, 9.8410072, -11.8911018, 9.8235149, -14.9903145, 15.0062370
13: -18.5415916, 6.7053633, -18.5436707, 6.7078047, -16.5865059, 16.5603104
14: 4.9827547, 36.3963318, 4.9766302, 36.3856201, -26.6628876, 26.6899567
15: -8.6535864, 9.2233200, -8.6817713, 9.2477541, -17.9013405, 17.9050903
16: -16.7216206, 2.5458734, -16.7120399, 2.5444808, -14.7772713, 14.7900734
17: 6.2231369, 30.6513252, 6.2122145, 30.6360359, -17.1967659, 17.1974831
18: -14.3659172, 5.1140051, -14.3748465, 5.1220961, -14.3680344, 14.3785591
19: -20.2623825, -4.3366170, -20.2676964, -4.3307643, -14.5143051, 14.5178185
20: -2.4078269, 11.2084713, -2.4089713, 11.2109795, -12.5973511, 12.5943604
21: -11.0640163, 3.2586622, -11.0672455, 3.2494411, -14.3134575, 14.3259077
22: -3.6891708, 13.1010571, -3.6986139, 13.0870438, -14.9067841, 14.9166908
23: -14.5538473, 0.3049612, -14.5775795, 0.3203793, -14.2659416, 14.2822418
24: -19.9310112, -5.1210332, -19.9297791, -5.1174436, -9.2641602, 9.2554474
25: -5.4494390, 10.8584232, -5.4497089, 10.8455353, -13.7797432, 13.7800865
26: -20.9968605, 1.2012904, -21.0142288, 1.1811292, -19.3007202, 19.2941742
27: -16.0072441, 2.1642952, -15.9949589, 2.1630712, -13.1828957, 13.1847496
28: -12.7698822, 4.6024427, -12.7886448, 4.6139712, -17.3838539, 17.3910866
29: -5.5635986, 11.8758335, -5.5936317, 11.8620501, -14.8994141, 14.9127617
30: -10.0415993, 6.2160473, -10.0433064, 6.2014298, -13.5328636, 13.5379219
31: -10.9541378, 6.9489937, -10.9589863, 6.9492321, -14.6203880, 14.6249390
32: -24.8947124, -4.5878167, -24.8911190, -4.5839863, -13.2709045, 13.2480507
33: -69.2875671, -40.1292877, -69.2877655, -40.1121140, -16.5998383, 16.6065407
34: -53.7329292, -30.9242458, -53.7329254, -30.9178619, -14.1192589, 14.0897179
35: -47.8006744, -26.0798721, -47.8032913, -26.0701904, -12.9912643, 12.9810448
36: -42.8092537, -19.2891674, -42.8023376, -19.2887268, -15.0817299, 15.0577049
37: -86.6643448, -55.5558968, -86.6644287, -55.5510292, -18.9079285, 18.9037399
38: -52.9065514, -24.3613815, -52.9100380, -24.3419018, -18.3119965, 18.2925644
39: -76.5219193, -44.6434250, -76.5358963, -44.6270409, -16.0434036, 16.0445480
40: -67.2462463, -43.5278053, -67.2250137, -43.5303192, -14.2881165, 14.2804317
41: -55.4189758, -32.9610214, -55.4057083, -32.9691467, -16.6561432, 16.6503372
42: -29.4516258, -9.8988466, -29.4519539, -9.9023552, -17.2178116, 17.2378197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5263481, upper bound: 12.5162705
time: 6.58 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5680800, upper bound: 12.5162705
time: 7.59 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -12.1177263, 3.6585436, -12.1176243, 3.6675134, -13.8600960, 13.8629608
1: -3.6652257, 7.3920908, -3.6595325, 7.3975639, -8.4755287, 8.4821606
2: -0.7552023, 13.4321604, -0.7389870, 13.4338303, -13.4477158, 13.4351807
3: -1.1273299, 11.3082752, -1.1223699, 11.3121300, -12.0180054, 12.0108757
4: -11.0960217, 5.4666700, -11.0958509, 5.4847326, -14.6482468, 14.6413536
5: 1.8424306, 17.7480793, 1.8514280, 17.7498779, -15.9074478, 15.8966513
6: -39.8596725, -18.2636318, -39.8875885, -18.2227249, -15.1037750, 15.1135445
7: -3.5765927, 12.2467804, -3.5617750, 12.2583981, -13.6029663, 13.5924110
8: -6.7193203, 8.5655212, -6.7065048, 8.5692272, -12.1196671, 12.0835991
9: -4.7583375, 11.6867199, -4.7842159, 11.6979561, -12.9723511, 12.9761314
10: 1.3255281, 25.7398472, 1.3143888, 25.7384739, -20.8976669, 20.8969345
11: -11.4933739, 4.2863479, -11.5013580, 4.2862673, -15.7796412, 15.7877064
12: -11.8939314, 9.8417511, -11.9027548, 9.8281593, -15.0034523, 15.0117226
13: -18.5482178, 6.7071609, -18.5536842, 6.7130103, -16.6098213, 16.5559769
14: 4.9618940, 36.3971252, 4.9408207, 36.4008179, -26.7021027, 26.7049942
15: -8.6587219, 9.2256050, -8.6906996, 9.2533770, -17.9120979, 17.9163055
16: -16.7337837, 2.5464282, -16.7291679, 2.5421281, -14.7776833, 14.8069534
17: 6.2130513, 30.6519318, 6.1944880, 30.6445312, -17.2043762, 17.2084732
18: -14.3672047, 5.1162720, -14.3759022, 5.1258240, -14.3768864, 14.3807087
19: -20.2649860, -4.3332243, -20.2724342, -4.3260970, -14.5216942, 14.5273590
20: -2.4095633, 11.2172565, -2.4160402, 11.2245617, -12.6061745, 12.6072578
21: -11.0668173, 3.2598455, -11.0716314, 3.2505546, -14.3173714, 14.3314772
22: -3.6902020, 13.1052799, -3.7001162, 13.0915794, -14.9129639, 14.9240799
23: -14.5563221, 0.3087890, -14.5800714, 0.3257165, -14.2744789, 14.2913284
24: -19.9323692, -5.1203957, -19.9320564, -5.1167707, -9.2669067, 9.2589340
25: -5.4534531, 10.8589916, -5.4566998, 10.8481503, -13.7828674, 13.7851372
26: -20.9998093, 1.2028410, -21.0195312, 1.1841757, -19.3065720, 19.3016586
27: -16.0084152, 2.1732702, -16.0049248, 2.1776481, -13.1765366, 13.2051926
28: -12.7716675, 4.6088452, -12.7954988, 4.6246662, -17.3963337, 17.4043446
29: -5.5646763, 11.8772697, -5.5957212, 11.8638744, -14.9022141, 14.9238281
30: -10.0456543, 6.2166190, -10.0495958, 6.2035027, -13.5353203, 13.5431976
31: -10.9569664, 6.9521279, -10.9635201, 6.9526987, -14.6263199, 14.6317291
32: -24.8957443, -4.5743361, -24.9060059, -4.5616546, -13.2716408, 13.2744102
33: -69.2884979, -40.1200790, -69.2949677, -40.0971184, -16.6065903, 16.6083755
34: -53.7334747, -30.9124413, -53.7433243, -30.8977680, -14.1240463, 14.1132469
35: -47.8011169, -26.0739174, -47.8073387, -26.0613289, -12.9999046, 12.9917755
36: -42.8094101, -19.2786083, -42.8091698, -19.2715988, -15.0891075, 15.0747757
37: -86.6653595, -55.5515175, -86.6679764, -55.5438881, -18.9181824, 18.9065704
38: -52.9070816, -24.3468132, -52.9239349, -24.3182106, -18.3124847, 18.3221130
39: -76.5231323, -44.6400414, -76.5380859, -44.6217346, -16.0543976, 16.0488625
40: -67.2471161, -43.5194664, -67.2326431, -43.5164299, -14.2801819, 14.2993565
41: -55.4199371, -32.9487953, -55.4181519, -32.9484329, -16.6554375, 16.6769371
42: -29.4525795, -9.8880939, -29.4626560, -9.8841572, -17.2180023, 17.2581749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5691138, upper bound: 12.5107835
time: 8.00 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5691138, upper bound: 12.5525295
time: 11.54 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.1374588, 3.6678951, -12.1100607, 3.6600761, -13.8879700, 13.8594933
1: -3.6775949, 7.3890829, -3.6598587, 7.3785124, -8.4748611, 8.4734020
2: -0.7663720, 13.4232645, -0.7321310, 13.4114542, -13.4465866, 13.4240990
3: -1.1403705, 11.3033772, -1.1288192, 11.2913256, -12.0251656, 12.0010166
4: -11.1242342, 5.4757376, -11.1022873, 5.4646759, -14.6511612, 14.6515465
5: 1.8263769, 17.7421093, 1.8440828, 17.7338581, -15.9074812, 15.8957138
6: -39.9311905, -18.2326450, -39.9215813, -18.2399940, -15.0940895, 15.1788445
7: -3.6100678, 12.2452183, -3.5759592, 12.2303610, -13.6091537, 13.5807953
8: -6.7215681, 8.5647812, -6.6979728, 8.5573158, -12.1014862, 12.0952492
9: -4.7809296, 11.7163744, -4.7741394, 11.7033510, -12.9975700, 12.9723701
10: 1.3141856, 25.7411137, 1.3295398, 25.7185783, -20.8896637, 20.9024429
11: -11.5019474, 4.2877450, -11.4966145, 4.2854681, -15.7874155, 15.7843590
12: -11.8870697, 9.8568325, -11.8693962, 9.8145370, -14.9721985, 15.0180511
13: -18.5420685, 6.7336016, -18.5388260, 6.7261133, -16.6210251, 16.5354233
14: 4.9706564, 36.4325027, 5.0087986, 36.4100761, -26.7149353, 26.6555786
15: -8.6934299, 9.2898216, -8.6922398, 9.3044157, -17.9978447, 17.9820614
16: -16.7470474, 2.5239954, -16.7232933, 2.5097911, -14.7789116, 14.7948494
17: 6.2303119, 30.6767044, 6.2645955, 30.6396255, -17.1965370, 17.1802902
18: -14.3998222, 5.1251726, -14.3898134, 5.1185827, -14.3991737, 14.4062386
19: -20.2739410, -4.3244233, -20.2629261, -4.3260512, -14.5266876, 14.5242882
20: -2.4168396, 11.2191277, -2.4018326, 11.2075472, -12.5957642, 12.6008148
21: -11.0748920, 3.2629189, -11.0619202, 3.2425697, -14.3174620, 14.3248386
22: -3.6826956, 13.1417866, -3.6647120, 13.0982599, -14.9089622, 14.9341507
23: -14.5788879, 0.3475132, -14.5697680, 0.3454480, -14.3046341, 14.2950668
24: -19.9359741, -5.1117554, -19.9307594, -5.1070085, -9.2563858, 9.2666016
25: -5.4507823, 10.8861008, -5.4327269, 10.8621807, -13.7792549, 13.7898140
26: -20.9899635, 1.2539020, -20.9636383, 1.1881425, -19.2902946, 19.3000793
27: -16.0232658, 2.1744390, -15.9992809, 2.1666284, -13.1910934, 13.2033920
28: -12.7929096, 4.6440759, -12.7775879, 4.6347437, -17.4276543, 17.4216633
29: -5.5715942, 11.9237881, -5.5502539, 11.8703222, -14.9136124, 14.9277306
30: -10.0403080, 6.2214870, -10.0301304, 6.1966867, -13.5328865, 13.5435219
31: -10.9839268, 6.9502397, -10.9711752, 6.9499316, -14.6318092, 14.6370621
32: -24.9275646, -4.5684986, -24.9197445, -4.5802069, -13.2511406, 13.2915497
33: -69.3202438, -40.1094284, -69.2970657, -40.1296234, -16.6036453, 16.6417694
34: -53.7660522, -30.9108047, -53.7512207, -30.9299374, -14.0830078, 14.1464500
35: -47.8079300, -26.0703449, -47.8049088, -26.0680866, -12.9739685, 12.9883003
36: -42.7960014, -19.2846584, -42.7837219, -19.2941437, -15.0662994, 15.0627785
37: -86.6685944, -55.5526733, -86.6615143, -55.5589371, -18.8842049, 18.9177895
38: -52.9440346, -24.3294735, -52.9209442, -24.3390656, -18.3120461, 18.3337173
39: -76.5495911, -44.6308212, -76.5465393, -44.6390076, -16.0571861, 16.0809364
40: -67.2843552, -43.5369415, -67.2387085, -43.5568771, -14.2920074, 14.3451328
41: -55.4417839, -32.9586334, -55.4246101, -32.9762383, -16.6323891, 16.6887016
42: -29.4647217, -9.8957701, -29.4612694, -9.9000254, -17.2025452, 17.2555656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5253979, upper bound: 12.5149794
time: 6.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671297, upper bound: 12.5149794
time: 8.43 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.1389723, 3.6684356, -12.1124592, 3.6594992, -13.8884659, 13.8627625
1: -3.6816225, 7.3901548, -3.6651316, 7.3809404, -8.4807510, 8.4781342
2: -0.7670979, 13.4247084, -0.7330036, 13.4139614, -13.4532776, 13.4261703
3: -1.1410661, 11.3085337, -1.1309433, 11.3001223, -12.0319977, 12.0092106
4: -11.1253796, 5.4773664, -11.1042633, 5.4656973, -14.6606979, 14.6588326
5: 1.8257604, 17.7456856, 1.8451548, 17.7396698, -15.9139099, 15.9005308
6: -39.9319992, -18.2201233, -39.9334908, -18.2189846, -15.0894585, 15.2032471
7: -3.6110322, 12.2473307, -3.5779958, 12.2340288, -13.6072922, 13.5860405
8: -6.7234535, 8.5656872, -6.7005148, 8.5590296, -12.1143951, 12.0980492
9: -4.7897530, 11.7168341, -4.7879319, 11.7065935, -13.0081520, 12.9786911
10: 1.2981000, 25.7418900, 1.3022499, 25.7288570, -20.9162369, 20.9189529
11: -11.5063553, 4.2878718, -11.5033531, 4.2853098, -15.7916651, 15.7912254
12: -11.8942251, 9.8575935, -11.8810797, 9.8191586, -14.9853134, 15.0235291
13: -18.5487747, 6.7353544, -18.5488262, 6.7313333, -16.6443367, 16.5311546
14: 4.9497919, 36.4332809, 4.9730043, 36.4252396, -26.7541351, 26.6706161
15: -8.6985826, 9.2920837, -8.7011642, 9.3100615, -18.0086441, 17.9932480
16: -16.7592087, 2.5245864, -16.7404213, 2.5073731, -14.7793236, 14.8118210
17: 6.2202044, 30.6773071, 6.2467995, 30.6481247, -17.2041664, 17.1912842
18: -14.4011154, 5.1274414, -14.3908930, 5.1223087, -14.4080200, 14.4083977
19: -20.2765274, -4.3210392, -20.2676468, -4.3213973, -14.5340881, 14.5338326
20: -2.4185956, 11.2279091, -2.4089150, 11.2211170, -12.6045876, 12.6136971
21: -11.0776949, 3.2641258, -11.0663176, 3.2436981, -14.3213930, 14.3304434
22: -3.6836939, 13.1460209, -3.6662278, 13.1028309, -14.9151001, 14.9415855
23: -14.5813389, 0.3513432, -14.5722494, 0.3507657, -14.3131676, 14.3040924
24: -19.9372826, -5.1111078, -19.9330425, -5.1063499, -9.2591248, 9.2700920
25: -5.4548311, 10.8866291, -5.4396696, 10.8648186, -13.7823944, 13.7948608
26: -20.9929352, 1.2554488, -20.9688969, 1.1912394, -19.2961731, 19.3075943
27: -16.0243855, 2.1834254, -16.0092735, 2.1812224, -13.1847420, 13.2238388
28: -12.7946911, 4.6504574, -12.7844343, 4.6454291, -17.4401207, 17.4348907
29: -5.5726652, 11.9252644, -5.5523386, 11.8721495, -14.9163895, 14.9388275
30: -10.0443535, 6.2220688, -10.0364456, 6.1987782, -13.5353355, 13.5488129
31: -10.9867306, 6.9533501, -10.9757137, 6.9533768, -14.6377487, 14.6438637
32: -24.9285965, -4.5549755, -24.9346352, -4.5578899, -13.2518845, 13.3179016
33: -69.3211899, -40.1002197, -69.3042679, -40.1145935, -16.6103783, 16.6435890
34: -53.7665520, -30.8989735, -53.7615623, -30.9098511, -14.0878258, 14.1699677
35: -47.8083572, -26.0643597, -47.8089981, -26.0592785, -12.9826241, 12.9990387
36: -42.7962036, -19.2741127, -42.7905693, -19.2770233, -15.0736694, 15.0798645
37: -86.6696091, -55.5483627, -86.6650696, -55.5517273, -18.8944626, 18.9206314
38: -52.9445648, -24.3148804, -52.9348526, -24.3153286, -18.3125420, 18.3632736
39: -76.5507965, -44.6274338, -76.5486755, -44.6337585, -16.0681610, 16.0852509
40: -67.2851868, -43.5286446, -67.2463531, -43.5429802, -14.2841034, 14.3640537
41: -55.4427071, -32.9463959, -55.4370575, -32.9555740, -16.6316872, 16.7153206
42: -29.4656982, -9.8850145, -29.4719696, -9.8818502, -17.2027092, 17.2759171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683063, upper bound: 12.5109786
time: 12.05 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683063, upper bound: 12.5524477
time: 12.46 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -12.1392508, 3.6807692, -12.1205101, 3.6812186, -13.8962059, 13.8846626
1: -3.6778884, 7.4012818, -3.6641207, 7.3984613, -8.4800072, 8.4934654
2: -0.7676303, 13.4374199, -0.7445617, 13.4353094, -13.4595337, 13.4505692
3: -1.1411514, 11.3133240, -1.1285046, 11.3082790, -12.0390625, 12.0103321
4: -11.1249104, 5.4895420, -11.1110611, 5.4880433, -14.6634445, 14.6741295
5: 1.8250737, 17.7494545, 1.8404188, 17.7464523, -15.9213791, 15.9090357
6: -39.9329720, -18.2341766, -39.9182701, -18.2434139, -15.1131172, 15.1743965
7: -3.6107237, 12.2613697, -3.5784962, 12.2572775, -13.6190186, 13.6159859
8: -6.7222795, 8.5740824, -6.7049508, 8.5726309, -12.1191978, 12.0909691
9: -4.7821202, 11.7228346, -4.7766356, 11.7147980, -13.0134888, 12.9871292
10: 1.3121624, 25.7490292, 1.3261962, 25.7330875, -20.9061050, 20.9314117
11: -11.5051994, 4.2884550, -11.5015974, 4.2880292, -15.7932281, 15.7900524
12: -11.9060287, 9.8581524, -11.9011488, 9.8255005, -15.0022736, 15.0339584
13: -18.5461254, 6.7346859, -18.5450020, 6.7224698, -16.5972023, 16.5894547
14: 4.9498396, 36.4329872, 4.9729691, 36.4058800, -26.7164993, 26.7040482
15: -8.6896782, 9.2906742, -8.6843596, 9.2860994, -17.9757767, 17.9750328
16: -16.7483597, 2.5458226, -16.7246742, 2.5456553, -14.8138618, 14.8022652
17: 6.1981225, 30.6772995, 6.2101541, 30.6497688, -17.2367973, 17.2244263
18: -14.4031887, 5.1301212, -14.3959112, 5.1271305, -14.4096012, 14.4179878
19: -20.2812710, -4.3242364, -20.2757835, -4.3234892, -14.5431519, 14.5382805
20: -2.4273183, 11.2203217, -2.4193859, 11.2139282, -12.6118584, 12.6126556
21: -11.0843582, 3.2640200, -11.0782623, 3.2503309, -14.3346891, 14.3422823
22: -3.7051609, 13.1422882, -3.7016494, 13.1103306, -14.9504738, 14.9504242
23: -14.5858860, 0.3478270, -14.5825310, 0.3453853, -14.3211288, 14.3074341
24: -19.9360008, -5.1111159, -19.9315376, -5.1124101, -9.2671165, 9.2733612
25: -5.4622827, 10.8865786, -5.4525909, 10.8602448, -13.8073349, 13.8033180
26: -21.0219803, 1.2548835, -21.0173435, 1.2118766, -19.3478699, 19.3127747
27: -16.0245132, 2.1761048, -16.0006676, 2.1694679, -13.2177773, 13.2002716
28: -12.8013840, 4.6444545, -12.7931452, 4.6384735, -17.4398575, 17.4375992
29: -5.5988617, 11.9241314, -5.5960083, 11.8896656, -14.9614182, 14.9438782
30: -10.0507603, 6.2227340, -10.0473757, 6.2060504, -13.5528374, 13.5516090
31: -10.9858780, 6.9504504, -10.9752150, 6.9499350, -14.6526375, 14.6406021
32: -24.9287643, -4.5705051, -24.9102821, -4.5833273, -13.2718124, 13.2799835
33: -69.3208694, -40.0960236, -69.3069916, -40.1063957, -16.6117401, 16.6576462
34: -53.7668839, -30.9021416, -53.7532196, -30.9147682, -14.1155853, 14.1307793
35: -47.8091049, -26.0690384, -47.8079147, -26.0677071, -12.9892426, 12.9978638
36: -42.8089600, -19.2837734, -42.8063087, -19.2871284, -15.0907822, 15.0685692
37: -86.6725998, -55.5442429, -86.6696930, -55.5450287, -18.9149094, 18.9311829
38: -52.9524078, -24.3287296, -52.9354858, -24.3395042, -18.3261909, 18.3508682
39: -76.5502930, -44.6215515, -76.5512848, -44.6234360, -16.0755119, 16.0869141
40: -67.2856674, -43.5203171, -67.2442169, -43.5298920, -14.3265533, 14.3145084
41: -55.4424477, -32.9539604, -55.4176598, -32.9682350, -16.6650658, 16.6686668
42: -29.4653988, -9.8963766, -29.4586983, -9.9012871, -17.2347946, 17.2491188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 918

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5264733, upper bound: 12.5326666
time: 12.54 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682048, upper bound: 12.5326666
time: 12.70 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.1407633, 3.6812944, -12.1229477, 3.6806445, -13.8967018, 13.8879051
1: -3.6819389, 7.4023342, -3.6693697, 7.4008827, -8.4859085, 8.4981995
2: -0.7683474, 13.4388618, -0.7454094, 13.4378376, -13.4662247, 13.4526596
3: -1.1418525, 11.3184977, -1.1306216, 11.3170595, -12.0458908, 12.0185204
4: -11.1260710, 5.4911556, -11.1130276, 5.4890871, -14.6730194, 14.6813927
5: 1.8244677, 17.7530079, 1.8415151, 17.7522602, -15.9277925, 15.9114933
6: -39.9337692, -18.2216682, -39.9302025, -18.2223568, -15.1084900, 15.1987724
7: -3.6116879, 12.2634583, -3.5805514, 12.2609472, -13.6171799, 13.6212006
8: -6.7241936, 8.5750313, -6.7074866, 8.5743561, -12.1320992, 12.0937634
9: -4.7909203, 11.7232952, -4.7904525, 11.7180195, -13.0240364, 12.9934502
10: 1.2960839, 25.7498055, 1.2989078, 25.7433548, -20.9326401, 20.9478836
11: -11.5096121, 4.2885809, -11.5083275, 4.2878752, -15.7974873, 15.7969084
12: -11.9131536, 9.8589134, -11.9127645, 9.8301249, -15.0153923, 15.0394516
13: -18.5528011, 6.7364244, -18.5550613, 6.7276754, -16.6204834, 16.5851707
14: 4.9289894, 36.4337654, 4.9371872, 36.4211502, -26.7556915, 26.7191162
15: -8.6948071, 9.2929420, -8.6933022, 9.2917175, -17.9865246, 17.9862442
16: -16.7605286, 2.5464025, -16.7418404, 2.5432563, -14.8143158, 14.8192406
17: 6.1880579, 30.6778946, 6.1923900, 30.6582870, -17.2444382, 17.2354088
18: -14.4044914, 5.1323671, -14.3969688, 5.1308718, -14.4184494, 14.4201508
19: -20.2838745, -4.3208451, -20.2805290, -4.3188138, -14.5505676, 14.5478172
20: -2.4290717, 11.2291260, -2.4264674, 11.2275219, -12.6206818, 12.6255226
21: -11.0871420, 3.2652078, -11.0826588, 3.2514505, -14.3385925, 14.3478661
22: -3.7061510, 13.1465139, -3.7031019, 13.1148720, -14.9566307, 14.9578705
23: -14.5883093, 0.3516531, -14.5850105, 0.3507075, -14.3296509, 14.3165092
24: -19.9373760, -5.1104774, -19.9338226, -5.1117325, -9.2698708, 9.2768288
25: -5.4663048, 10.8871660, -5.4595366, 10.8628731, -13.8104935, 13.8083382
26: -21.0249481, 1.2564323, -21.0226498, 1.2149680, -19.3536987, 19.3202438
27: -16.0256786, 2.1850908, -16.0106277, 2.1840625, -13.2113991, 13.2206917
28: -12.8031464, 4.6508713, -12.7999544, 4.6491241, -17.4522705, 17.4508247
29: -5.5999451, 11.9255953, -5.5980768, 11.8915138, -14.9642410, 14.9549751
30: -10.0548086, 6.2232962, -10.0536623, 6.2081223, -13.5552711, 13.5568924
31: -10.9886608, 6.9535756, -10.9797268, 6.9533701, -14.6585541, 14.6474037
32: -24.9298172, -4.5570183, -24.9251442, -4.5610051, -13.2725830, 13.3063393
33: -69.3218231, -40.0867996, -69.3142166, -40.0913773, -16.6184540, 16.6594467
34: -53.7673759, -30.8903236, -53.7635803, -30.8946819, -14.1204033, 14.1543159
35: -47.8095016, -26.0630875, -47.8120193, -26.0588875, -12.9978943, 13.0086021
36: -42.8091240, -19.2732277, -42.8132019, -19.2699852, -15.0981483, 15.0855865
37: -86.6736145, -55.5398788, -86.6732025, -55.5378418, -18.9251747, 18.9339371
38: -52.9529343, -24.3141670, -52.9493370, -24.3157768, -18.3266830, 18.3804321
39: -76.5515213, -44.6181946, -76.5533905, -44.6181259, -16.0865211, 16.0912132
40: -67.2864838, -43.5119781, -67.2518158, -43.5159988, -14.3186340, 14.3334045
41: -55.4433289, -32.9417152, -55.4300919, -32.9474869, -16.6643448, 16.6952896
42: -29.4663792, -9.8856173, -29.4694118, -9.8831148, -17.2349777, 17.2694626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5692372, upper bound: 12.5275078
time: 6.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5692372, upper bound: 12.5692367
time: 10.14 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.89 seconds
IS_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.4567553, upper bound: 12.5673752
IS_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.4975104, upper bound: 12.5673752
IS_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.4946436, upper bound: 12.5684410
IS_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5355290, upper bound: 12.5684410
IS_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5252697, upper bound: 12.4991007
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5670018, upper bound: 12.4991006
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5681810, upper bound: 12.4943138
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5681810, upper bound: 12.5360208
IS_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5263481, upper bound: 12.5162705
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5680800, upper bound: 12.5162705
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5691138, upper bound: 12.5107835
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5691138, upper bound: 12.5525295
IS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5253979, upper bound: 12.5149794
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5671297, upper bound: 12.5149794
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5683063, upper bound: 12.5109786
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5683063, upper bound: 12.5524477
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5264733, upper bound: 12.5326666
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5682048, upper bound: 12.5326666
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5692372, upper bound: 12.5275078
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 18.89
Output dim: 14, lower bound: -12.5692372, upper bound: 12.5692367

## BFS IS instance: IS_A1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -12.1176109, 3.6716638, -12.1207485, 3.6761119, -13.8656883, 13.8767128
1: -3.6616158, 7.3896427, -3.6651030, 7.3959646, -8.4632301, 8.4847355
2: -0.7410105, 13.4214001, -0.7435292, 13.4310684, -13.4293823, 13.4342613
3: -1.1262919, 11.2921543, -1.1294695, 11.3050900, -12.0138855, 11.9936523
4: -11.1078758, 5.4775820, -11.1114483, 5.4833393, -14.6440887, 14.6613464
5: 1.8429861, 17.7325783, 1.8428173, 17.7435398, -15.9005537, 15.8897610
6: -39.9151115, -18.2484589, -39.9292488, -18.2347050, -15.1110344, 15.1656914
7: -3.5758405, 12.2408018, -3.5788081, 12.2514277, -13.5646591, 13.6028175
8: -6.6996446, 8.5625191, -6.7041092, 8.5693226, -12.0956497, 12.0780106
9: -4.7650270, 11.7116356, -4.7802086, 11.7163134, -12.9933205, 12.9840050
10: 1.3511081, 25.7304268, 1.3241282, 25.7415562, -20.8792801, 20.9173126
11: -11.4893417, 4.2860589, -11.4997311, 4.2870331, -15.7763748, 15.7857895
12: -11.8801956, 9.8223696, -11.8976498, 9.8282413, -14.9816780, 14.9865494
13: -18.5469685, 6.7183580, -18.5564842, 6.7252803, -16.5726624, 16.6177711
14: 5.0103474, 36.4042816, 4.9728441, 36.4200592, -26.6730042, 26.6833267
15: -8.6855860, 9.2802458, -8.6904783, 9.2879591, -17.9735451, 17.9707241
16: -16.7094727, 2.5312834, -16.7250443, 2.5350130, -14.7758484, 14.7973480
17: 6.2340455, 30.6478519, 6.2120485, 30.6570683, -17.1987152, 17.2066078
18: -14.3902035, 5.1172571, -14.3935738, 5.1252871, -14.4002228, 14.4017792
19: -20.2680893, -4.3258214, -20.2756500, -4.3219118, -14.5262184, 14.5258446
20: -2.4138348, 11.2120342, -2.4225373, 11.2188282, -12.6049118, 12.6037369
21: -11.0636997, 3.2494252, -11.0735950, 3.2504804, -14.3141804, 14.3230200
22: -3.6834857, 13.1055107, -3.6938932, 13.1096964, -14.9319000, 14.9020271
23: -14.5751057, 0.3423319, -14.5794840, 0.3469901, -14.3152237, 14.2907715
24: -19.9304523, -5.1142960, -19.9326782, -5.1126552, -9.2571793, 9.2631912
25: -5.4370661, 10.8585892, -5.4470458, 10.8620548, -13.7876587, 13.7694931
26: -20.9900723, 1.2085645, -21.0076141, 1.2126019, -19.3223724, 19.2575378
27: -15.9983444, 2.1598167, -16.0085602, 2.1717985, -13.2111053, 13.1909065
28: -12.7855816, 4.6339779, -12.7953243, 4.6416826, -17.4272652, 17.4293022
29: -5.5782652, 11.8875637, -5.5895853, 11.8899498, -14.9399071, 14.9025726
30: -10.0351706, 6.2044964, -10.0458078, 6.2071462, -13.5372543, 13.5339508
31: -10.9676037, 6.9451780, -10.9743395, 6.9496441, -14.6417313, 14.6318550
32: -24.9072571, -4.5890656, -24.9234982, -4.5768118, -13.2618713, 13.2710266
33: -69.3041306, -40.1246414, -69.3125610, -40.1081200, -16.6082077, 16.6309204
34: -53.7510948, -30.9375877, -53.7626686, -30.9166775, -14.1073227, 14.1049080
35: -47.8145332, -26.0772133, -47.8189774, -26.0685902, -12.9831009, 12.9862061
36: -42.8127289, -19.3007889, -42.8215752, -19.2839851, -15.0865021, 15.0594978
37: -86.6689987, -55.5560379, -86.6753464, -55.5455933, -18.9039001, 18.9039192
38: -52.9321976, -24.3543682, -52.9475861, -24.3360252, -18.3085938, 18.3397827
39: -76.5537872, -44.6317558, -76.5582275, -44.6249008, -16.0675163, 16.0673065
40: -67.2418518, -43.5552826, -67.2509155, -43.5374146, -14.3289833, 14.3051453
41: -55.4153214, -32.9884491, -55.4287033, -32.9698906, -16.6675644, 16.6459274
42: -29.4561386, -9.8969460, -29.4682064, -9.8871956, -17.2464752, 17.2243233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4220940, upper bound: 12.5657701
time: 6.46 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4552077, upper bound: 12.5657701
time: 32.34 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -12.1163082, 3.6771178, -12.1192045, 3.6763697, -13.8656273, 13.8816147
1: -3.6572032, 7.3932643, -3.6618633, 7.3962431, -8.4654160, 8.4963531
2: -0.7399206, 13.4233646, -0.7422610, 13.4312801, -13.4300156, 13.4362755
3: -1.1263638, 11.2897768, -1.1294303, 11.3019695, -12.0198441, 11.9954643
4: -11.1029749, 5.4785705, -11.1077299, 5.4836035, -14.6418991, 14.6613235
5: 1.8425422, 17.7325096, 1.8428640, 17.7424355, -15.8998928, 15.8896456
6: -39.9223595, -18.2549534, -39.9293823, -18.2395535, -15.1153183, 15.1651878
7: -3.5773261, 12.2440453, -3.5782139, 12.2514534, -13.5655136, 13.6123543
8: -6.6991720, 8.5635872, -6.7023268, 8.5695229, -12.0964622, 12.0796947
9: -4.7728176, 11.7318735, -4.7822027, 11.7162781, -12.9977150, 12.9980125
10: 1.3370781, 25.7496300, 1.3198891, 25.7414474, -20.8909912, 20.9370499
11: -11.4966927, 4.2920122, -11.5013962, 4.2864137, -15.7831059, 15.7934084
12: -11.8858938, 9.8273621, -11.8978329, 9.8279505, -14.9866829, 14.9904251
13: -18.5495911, 6.7290487, -18.5568695, 6.7251863, -16.5746002, 16.6217155
14: 4.9919567, 36.4422569, 4.9677134, 36.4201279, -26.6882629, 26.7188797
15: -8.6828356, 9.2845240, -8.6877432, 9.2884645, -17.9713001, 17.9722672
16: -16.7200050, 2.5441310, -16.7267036, 2.5351019, -14.7831268, 14.8161926
17: 6.2256174, 30.6545277, 6.2111678, 30.6571331, -17.2123947, 17.2045746
18: -14.3864946, 5.1189370, -14.3897676, 5.1255693, -14.4031563, 14.4041481
19: -20.2696724, -4.3246565, -20.2743607, -4.3216720, -14.5318680, 14.5255623
20: -2.4183879, 11.2109890, -2.4227333, 11.2177029, -12.6063271, 12.6034622
21: -11.0657291, 3.2500608, -11.0718107, 3.2504199, -14.3161488, 14.3218718
22: -3.6900411, 13.1097488, -3.6937685, 13.1102047, -14.9439812, 14.9054642
23: -14.5795679, 0.3436172, -14.5798769, 0.3470039, -14.3177414, 14.2928543
24: -19.9290924, -5.1139212, -19.9311333, -5.1127076, -9.2587357, 9.2646103
25: -5.4393258, 10.8600607, -5.4466810, 10.8620749, -13.7950211, 13.7688980
26: -20.9890862, 1.2106719, -21.0044479, 1.2127209, -19.3261337, 19.2583618
27: -16.0082378, 2.1633062, -16.0084095, 2.1728337, -13.2125931, 13.1947632
28: -12.7939043, 4.6372995, -12.7955074, 4.6424942, -17.4363976, 17.4328079
29: -5.5805893, 11.8891706, -5.5896597, 11.8896523, -14.9472160, 14.9034576
30: -10.0403805, 6.2109766, -10.0467587, 6.2068539, -13.5419006, 13.5419884
31: -10.9753056, 6.9472513, -10.9741230, 6.9502239, -14.6483383, 14.6333237
32: -24.9171925, -4.5978527, -24.9236069, -4.5832968, -13.2717819, 13.2708206
33: -69.3357620, -40.1112976, -69.3122711, -40.1037254, -16.6416168, 16.6395798
34: -53.7785072, -30.9265995, -53.7626762, -30.9135590, -14.1338806, 14.1134300
35: -47.8321037, -26.0696716, -47.8189049, -26.0667763, -13.0026321, 12.9931602
36: -42.8332520, -19.2945862, -42.8214302, -19.2826080, -15.1029968, 15.0633430
37: -86.6796799, -55.5505524, -86.6750793, -55.5439835, -18.9111671, 18.9082298
38: -52.9587784, -24.3466072, -52.9475021, -24.3340874, -18.3238831, 18.3445663
39: -76.5703430, -44.6249771, -76.5580063, -44.6225853, -16.0836983, 16.0726471
40: -67.2628632, -43.5482407, -67.2507477, -43.5345306, -14.3274498, 14.3096199
41: -55.4391251, -32.9802094, -55.4287872, -32.9673309, -16.6795959, 16.6489792
42: -29.4623451, -9.9090633, -29.4684219, -9.8953180, -17.2504578, 17.2223473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4630811, upper bound: 12.5657701
time: 10.63 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4959760, upper bound: 12.5657701
time: 11.10 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.1200457, 3.6710749, -12.1222506, 3.6766448, -13.8689232, 13.8772011
1: -3.6668787, 7.3920765, -3.6691418, 7.3970165, -8.4679718, 8.4906387
2: -0.7418890, 13.4238939, -0.7442492, 13.4325352, -13.4314461, 13.4409637
3: -1.1284084, 11.3009386, -1.1301800, 11.3102417, -12.0220833, 12.0004807
4: -11.1098442, 5.4785938, -11.1126280, 5.4849429, -14.6513367, 14.6708908
5: 1.8440561, 17.7383728, 1.8421860, 17.7470989, -15.9030428, 15.8961868
6: -39.9270248, -18.2274418, -39.9300690, -18.2222137, -15.1354141, 15.1610718
7: -3.5778763, 12.2444611, -3.5797822, 12.2535238, -13.5699005, 13.6009521
8: -6.7021480, 8.5642376, -6.7060256, 8.5702610, -12.0984192, 12.0909309
9: -4.7788424, 11.7148457, -4.7890563, 11.7167816, -12.9996185, 12.9945679
10: 1.3238416, 25.7406845, 1.3080649, 25.7423172, -20.8957825, 20.9438400
11: -11.4960728, 4.2859139, -11.5041180, 4.2871261, -15.7831993, 15.7900314
12: -11.8918610, 9.8269939, -11.9047689, 9.8289785, -14.9871712, 14.9996872
13: -18.5569935, 6.7236032, -18.5631905, 6.7270689, -16.5683899, 16.6410866
14: 4.9745398, 36.4194717, 4.9519720, 36.4208107, -26.6880951, 26.7225647
15: -8.6945009, 9.2858944, -8.6956263, 9.2902822, -17.9847832, 17.9815216
16: -16.7266006, 2.5288906, -16.7372208, 2.5355687, -14.7928047, 14.7977715
17: 6.2162757, 30.6563435, 6.2020078, 30.6576977, -17.2097130, 17.2142258
18: -14.3912678, 5.1209855, -14.3948669, 5.1275558, -14.4023952, 14.4106140
19: -20.2728004, -4.3211608, -20.2782249, -4.3185282, -14.5357475, 14.5332184
20: -2.4209213, 11.2256432, -2.4242840, 11.2276411, -12.6178169, 12.6125603
21: -11.0680771, 3.2505341, -11.0763779, 3.2516747, -14.3197517, 14.3269119
22: -3.6849551, 13.1100750, -3.6949050, 13.1139297, -14.9392891, 14.9081841
23: -14.5775948, 0.3476536, -14.5819092, 0.3508058, -14.3242912, 14.2992783
24: -19.9327030, -5.1136332, -19.9340172, -5.1120028, -9.2606544, 9.2659264
25: -5.4440012, 10.8612146, -5.4510651, 10.8626032, -13.7926826, 13.7726021
26: -20.9953651, 1.2116404, -21.0105629, 1.2141695, -19.3298950, 19.2633667
27: -16.0083408, 2.1743836, -16.0096970, 2.1807551, -13.2315369, 13.1845551
28: -12.7924347, 4.6446362, -12.7970924, 4.6480780, -17.4405136, 17.4417286
29: -5.5803938, 11.8894253, -5.5906563, 11.8914194, -14.9509735, 14.9053612
30: -10.0414820, 6.2065630, -10.0498981, 6.2076931, -13.5425110, 13.5363960
31: -10.9721584, 6.9486284, -10.9771404, 6.9527631, -14.6485138, 14.6377678
32: -24.9221230, -4.5667830, -24.9245262, -4.5633192, -13.2882538, 13.2717896
33: -69.3113861, -40.1096153, -69.3135071, -40.0988770, -16.6099930, 16.6376572
34: -53.7614479, -30.9174728, -53.7631760, -30.9048634, -14.1308517, 14.1096916
35: -47.8186035, -26.0683937, -47.8193970, -26.0626259, -12.9938202, 12.9948425
36: -42.8195915, -19.2835808, -42.8217010, -19.2733955, -15.1035614, 15.0668297
37: -86.6725388, -55.5489197, -86.6763687, -55.5412598, -18.9067307, 18.9141541
38: -52.9460564, -24.3306465, -52.9480858, -24.3214645, -18.3381462, 18.3402710
39: -76.5559998, -44.6264725, -76.5594254, -44.6215591, -16.0718117, 16.0783005
40: -67.2494583, -43.5413818, -67.2518005, -43.5291100, -14.3478966, 14.2972412
41: -55.4278030, -32.9677582, -55.4296341, -32.9576759, -16.6941490, 16.6452446
42: -29.4668579, -9.8787670, -29.4691887, -9.8764324, -17.2667961, 17.2245140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4592958, upper bound: 12.5669280
time: 7.01 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4931458, upper bound: 12.5669280
time: 11.88 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -12.1187201, 3.6765299, -12.1207104, 3.6768799, -13.8689003, 13.8820992
1: -3.6624565, 7.3956637, -3.6658874, 7.3973083, -8.4701500, 8.5022354
2: -0.7408262, 13.4258757, -0.7429492, 13.4327602, -13.4320679, 13.4429321
3: -1.1284873, 11.2985535, -1.1301438, 11.3071404, -12.0280571, 12.0022774
4: -11.1049557, 5.4796171, -11.1089039, 5.4852304, -14.6491852, 14.6709023
5: 1.8436246, 17.7383175, 1.8422060, 17.7460155, -15.9023914, 15.8961115
6: -39.9343033, -18.2339153, -39.9301682, -18.2270451, -15.1396866, 15.1605415
7: -3.5793664, 12.2477226, -3.5792074, 12.2535572, -13.5707474, 13.6104965
8: -6.7016959, 8.5653124, -6.7042475, 8.5704222, -12.0992622, 12.0926132
9: -4.7866230, 11.7351189, -4.7910347, 11.7167339, -13.0040092, 13.0085793
10: 1.3097458, 25.7598953, 1.3038368, 25.7422009, -20.9074860, 20.9635544
11: -11.5034180, 4.2918582, -11.5057898, 4.2865267, -15.7899446, 15.7976475
12: -11.8975735, 9.8319883, -11.9049530, 9.8286781, -14.9921684, 15.0035477
13: -18.5596924, 6.7342472, -18.5635567, 6.7269726, -16.5703316, 16.6450272
14: 4.9561663, 36.4574814, 4.9468317, 36.4209099, -26.7032928, 26.7580643
15: -8.6917477, 9.2901850, -8.6929045, 9.2907696, -17.9825172, 17.9830894
16: -16.7371235, 2.5417655, -16.7388706, 2.5356708, -14.8000374, 14.8166237
17: 6.2078447, 30.6630039, 6.2010870, 30.6577129, -17.2233849, 17.2121811
18: -14.3875494, 5.1226420, -14.3910418, 5.1278448, -14.4053307, 14.4129868
19: -20.2743759, -4.3200040, -20.2769394, -4.3182878, -14.5414085, 14.5329552
20: -2.4254794, 11.2245684, -2.4244778, 11.2264881, -12.6192284, 12.6122742
21: -11.0701351, 3.2511859, -11.0746069, 3.2515974, -14.3217325, 14.3257923
22: -3.6915176, 13.1142883, -3.6947958, 13.1144304, -14.9514275, 14.9116249
23: -14.5820923, 0.3489261, -14.5822916, 0.3508103, -14.3268166, 14.3013649
24: -19.9313927, -5.1132803, -19.9324875, -5.1120586, -9.2622147, 9.2673531
25: -5.4462790, 10.8626862, -5.4507046, 10.8626194, -13.8000717, 13.7720108
26: -20.9943790, 1.2137558, -21.0074196, 1.2142785, -19.3336487, 19.2641907
27: -16.0182228, 2.1778884, -16.0095787, 2.1817911, -13.2330284, 13.1883850
28: -12.8007469, 4.6479964, -12.7972488, 4.6488953, -17.4496422, 17.4452457
29: -5.5826812, 11.8910217, -5.5907440, 11.8910828, -14.9583015, 14.9062843
30: -10.0466843, 6.2130184, -10.0508032, 6.2074103, -13.5471878, 13.5444221
31: -10.9798584, 6.9507017, -10.9769325, 6.9533491, -14.6551285, 14.6392479
32: -24.9320621, -4.5755386, -24.9246216, -4.5698180, -13.2981224, 13.2715874
33: -69.3430328, -40.0962753, -69.3132019, -40.0944977, -16.6434097, 16.6463356
34: -53.7888489, -30.9064884, -53.7632141, -30.9017601, -14.1574211, 14.1182289
35: -47.8361435, -26.0608463, -47.8193130, -26.0608711, -13.0133705, 13.0018120
36: -42.8401108, -19.2774334, -42.8216095, -19.2720108, -15.1200638, 15.0706940
37: -86.6832123, -55.5434036, -86.6761017, -55.5396118, -18.9139938, 18.9184685
38: -52.9725571, -24.3229313, -52.9480286, -24.3195248, -18.3534393, 18.3450775
39: -76.5724564, -44.6196899, -76.5591736, -44.6192169, -16.0879936, 16.0836411
40: -67.2705002, -43.5343399, -67.2516098, -43.5262108, -14.3463593, 14.3017254
41: -55.4515572, -32.9595070, -55.4297028, -32.9551201, -16.7062225, 16.6482697
42: -29.4730701, -9.8908815, -29.4693985, -9.8845634, -17.2707977, 17.2225037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5006482, upper bound: 12.5669280
time: 26.13 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5340302, upper bound: 12.5669280
time: 14.82 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -12.1117802, 3.6494076, -12.1026001, 3.6466589, -13.8491325, 13.8374901
1: -3.6548307, 7.3811359, -3.6460314, 7.3749084, -8.4622917, 8.4659023
2: -0.7511145, 13.4170799, -0.7239038, 13.4070368, -13.4266129, 13.4065361
3: -1.1254212, 11.2867489, -1.1203288, 11.2814665, -12.0002747, 11.9893494
4: -11.0876122, 5.4506168, -11.0805902, 5.4598656, -14.6215668, 14.6091232
5: 1.8443346, 17.7351818, 1.8541803, 17.7295074, -15.8851728, 15.8810015
6: -39.8631668, -18.2863731, -39.8785133, -18.2476158, -15.0901070, 15.0876160
7: -3.5753562, 12.2309017, -3.5561163, 12.2274532, -13.5927811, 13.5593719
8: -6.7139726, 8.5552969, -6.6942110, 8.5519009, -12.0865784, 12.0840302
9: -4.7480097, 11.6994247, -4.7662401, 11.6829567, -12.9427567, 12.9652710
10: 1.3420992, 25.7499180, 1.3464398, 25.7133923, -20.8543701, 20.8654633
11: -11.4867992, 4.2904816, -11.4885216, 4.2828026, -15.7696018, 15.7790031
12: -11.8699989, 9.8436718, -11.8579483, 9.8118439, -14.9612427, 14.9916992
13: -18.5367966, 6.7133141, -18.5363541, 6.7106628, -16.6090431, 16.5077400
14: 5.0030794, 36.4331856, 5.0153589, 36.3895607, -26.6582794, 26.6682053
15: -8.6521139, 9.2244911, -8.6857986, 9.2654552, -17.9175682, 17.9102898
16: -16.7212887, 2.5363955, -16.7079544, 2.5084374, -14.7380600, 14.7955589
17: 6.2509456, 30.6567955, 6.2676158, 30.6256580, -17.1661110, 17.1476021
18: -14.3563290, 5.1091280, -14.3638029, 5.1131220, -14.3548279, 14.3651161
19: -20.2541199, -4.3363352, -20.2524757, -4.3334026, -14.4985504, 14.4989662
20: -2.4001775, 11.2047863, -2.3908429, 11.2027988, -12.5804405, 12.5802307
21: -11.0523338, 3.2579799, -11.0472269, 3.2415078, -14.2938414, 14.3052063
22: -3.6719339, 13.1018753, -3.6609931, 13.0741682, -14.8740692, 14.8991661
23: -14.5493755, 0.3037543, -14.5643206, 0.3194675, -14.2486801, 14.2684746
24: -19.9284382, -5.1217241, -19.9269161, -5.1122842, -9.2528000, 9.2484932
25: -5.4379158, 10.8586016, -5.4284430, 10.8471909, -13.7558556, 13.7626038
26: -20.9595737, 1.2016940, -20.9553566, 1.1571615, -19.2408066, 19.2775650
27: -16.0147552, 2.1630359, -15.9929199, 2.1598818, -13.1548691, 13.1885414
28: -12.7685041, 4.6023293, -12.7727032, 4.6096830, -17.3781872, 17.3750324
29: -5.5376844, 11.8764315, -5.5475264, 11.8420839, -14.8567963, 14.8948860
30: -10.0328445, 6.2206068, -10.0253925, 6.1914873, -13.5139847, 13.5356598
31: -10.9565125, 6.9487028, -10.9532080, 6.9488487, -14.6017227, 14.6192589
32: -24.9021568, -4.5996943, -24.9000778, -4.5896587, -13.2561150, 13.2530518
33: -69.3172302, -40.1420784, -69.2769165, -40.1366653, -16.6187553, 16.5871315
34: -53.7587509, -30.9363823, -53.7306290, -30.9365730, -14.1061287, 14.0993767
35: -47.8165894, -26.0810623, -47.7999725, -26.0720539, -12.9921417, 12.9711456
36: -42.8162537, -19.2921619, -42.7793732, -19.2980061, -15.0704155, 15.0488510
37: -86.6695862, -55.5638161, -86.6553650, -55.5654678, -18.8810310, 18.8897820
38: -52.9237366, -24.3650475, -52.8950081, -24.3442688, -18.3078423, 18.2704926
39: -76.5356598, -44.6526031, -76.5300751, -44.6432800, -16.0355606, 16.0361023
40: -67.2649994, -43.5447960, -67.2189178, -43.5577393, -14.2482414, 14.3107128
41: -55.4410248, -32.9676781, -55.4122124, -32.9792213, -16.6311684, 16.6656761
42: -29.4557686, -9.9148521, -29.4541111, -9.9112339, -17.1860580, 17.2365646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5653244, upper bound: 12.4628685
time: 8.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5653244, upper bound: 12.4974587
time: 6.88 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.1153259, 3.6451430, -12.1058264, 3.6451983, -13.8498878, 13.8356171
1: -3.6641848, 7.3793087, -3.6536341, 7.3763385, -8.4673615, 8.4577179
2: -0.7534691, 13.4173708, -0.7254570, 13.4085579, -13.4326820, 13.4066162
3: -1.1263157, 11.2964993, -1.1221936, 11.2911501, -11.9982529, 11.9986076
4: -11.0945492, 5.4521255, -11.0854454, 5.4597187, -14.6335907, 14.6161613
5: 1.8439078, 17.7398720, 1.8554420, 17.7353592, -15.8914509, 15.8844299
6: -39.8573532, -18.2645321, -39.8896141, -18.2246170, -15.0792084, 15.1144562
7: -3.5754330, 12.2302608, -3.5581234, 12.2306128, -13.5909348, 13.5541687
8: -6.7175908, 8.5557280, -6.6972656, 8.5528927, -12.0992165, 12.0845280
9: -4.7535005, 11.6799879, -4.7735896, 11.6858816, -12.9526596, 12.9538193
10: 1.3332129, 25.7317085, 1.3302417, 25.7234840, -20.8756104, 20.8558655
11: -11.4873180, 4.2851977, -11.4901581, 4.2827148, -15.7700329, 15.7753563
12: -11.8733978, 9.8399963, -11.8675013, 9.8162146, -14.9708862, 14.9918175
13: -18.5426826, 6.7053986, -18.5441399, 6.7150607, -16.6311569, 16.4987335
14: 4.9907217, 36.3963089, 4.9945412, 36.4042816, -26.6917801, 26.6382217
15: -8.6613817, 9.2236605, -8.6960735, 9.2692451, -17.9306259, 17.9197350
16: -16.7281055, 2.5243621, -16.7182236, 2.5057049, -14.7368622, 14.7878723
17: 6.2470541, 30.6510849, 6.2530036, 30.6338100, -17.1604118, 17.1602859
18: -14.3627062, 5.1106434, -14.3672829, 5.1156759, -14.3623943, 14.3632431
19: -20.2565269, -4.3337049, -20.2571030, -4.3293753, -14.5006790, 14.5084457
20: -2.3983219, 11.2154293, -2.3967738, 11.2167501, -12.5880699, 12.5931931
21: -11.0554409, 3.2586584, -11.0510502, 3.2425809, -14.2980213, 14.3097086
22: -3.6671371, 13.1034679, -3.6618528, 13.0766249, -14.8666725, 14.9045181
23: -14.5484409, 0.3074946, -14.5653543, 0.3235965, -14.2544861, 14.2756271
24: -19.9317665, -5.1212201, -19.9301033, -5.1117935, -9.2545547, 9.2499695
25: -5.4409471, 10.8581181, -5.4345064, 10.8493233, -13.7513657, 13.7684326
26: -20.9658184, 1.2015603, -20.9615173, 1.1597085, -19.2442207, 19.2829208
27: -16.0066185, 2.1702089, -16.0024071, 2.1716981, -13.1466751, 13.2054672
28: -12.7626305, 4.6070709, -12.7786942, 4.6178622, -17.3804932, 17.3857651
29: -5.5369930, 11.8765945, -5.5490408, 11.8438702, -14.8517838, 14.9055634
30: -10.0336266, 6.2150774, -10.0288754, 6.1934786, -13.5131760, 13.5314865
31: -10.9534798, 6.9509497, -10.9560547, 6.9505424, -14.6019173, 14.6236725
32: -24.8939629, -4.5745869, -24.9141312, -4.5636230, -13.2445679, 13.2820320
33: -69.2872467, -40.1391830, -69.2836990, -40.1330147, -16.5862122, 16.5861282
34: -53.7322617, -30.9276886, -53.7404938, -30.9273720, -14.0768814, 14.1217957
35: -47.7997131, -26.0785561, -47.8038788, -26.0691185, -12.9773598, 12.9788055
36: -42.7962265, -19.2832394, -42.7861023, -19.2868061, -15.0576820, 15.0656738
37: -86.6607361, -55.5622177, -86.6584473, -55.5626678, -18.8826752, 18.8896713
38: -52.8982239, -24.3523293, -52.9083862, -24.3283310, -18.2887077, 18.2996521
39: -76.5214920, -44.6523285, -76.5313568, -44.6440125, -16.0282669, 16.0371437
40: -67.2453308, -43.5394516, -67.2261810, -43.5508156, -14.2408028, 14.3261909
41: -55.4187393, -32.9580307, -55.4239464, -32.9666786, -16.6149788, 16.6926270
42: -29.4512691, -9.8895168, -29.4638577, -9.8874178, -17.1800385, 17.2610550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4574710
time: 7.09 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4927019
time: 11.75 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -12.1137915, 3.6453900, -12.1045055, 3.6506085, -13.8548050, 13.8355751
1: -3.6609495, 7.3796091, -3.6492186, 7.3799515, -8.4789391, 8.4598942
2: -0.7522135, 13.4175892, -0.7244006, 13.4105167, -13.4346809, 13.4072342
3: -1.1262770, 11.2934027, -1.1222743, 11.2887774, -12.0000687, 12.0045929
4: -11.0908241, 5.4524117, -11.0805426, 5.4607248, -14.6335526, 14.6140099
5: 1.8439436, 17.7387753, 1.8550196, 17.7353077, -15.8913641, 15.8837557
6: -39.8574829, -18.2693443, -39.8968811, -18.2310925, -15.0786934, 15.1187553
7: -3.5748668, 12.2302818, -3.5596030, 12.2338829, -13.6004791, 13.5550385
8: -6.7158041, 8.5558910, -6.6968145, 8.5539694, -12.1009064, 12.0853462
9: -4.7555060, 11.6799164, -4.7813959, 11.7061443, -12.9666672, 12.9582329
10: 1.3289762, 25.7316132, 1.3161759, 25.7426796, -20.8953094, 20.8675690
11: -11.4889812, 4.2845731, -11.4974890, 4.2886615, -15.7776432, 15.7820625
12: -11.8735876, 9.8396988, -11.8731585, 9.8212166, -14.9747696, 14.9968033
13: -18.5430832, 6.7052879, -18.5467911, 6.7256670, -16.6351357, 16.5006828
14: 4.9855814, 36.3963737, 4.9761648, 36.4422989, -26.7273102, 26.6534424
15: -8.6586685, 9.2241526, -8.6933270, 9.2735834, -17.9322510, 17.9174805
16: -16.7297344, 2.5244484, -16.7287350, 2.5185699, -14.7557144, 14.7951279
17: 6.2461371, 30.6511002, 6.2445788, 30.6404457, -17.1583672, 17.1739273
18: -14.3589048, 5.1109319, -14.3635845, 5.1173420, -14.3647804, 14.3661842
19: -20.2552643, -4.3334417, -20.2586899, -4.3282170, -14.5003662, 14.5140686
20: -2.3985152, 11.2142906, -2.4013281, 11.2156887, -12.5877838, 12.5946007
21: -11.0536699, 3.2585855, -11.0531063, 3.2432060, -14.2968760, 14.3116913
22: -3.6670189, 13.1039810, -3.6684084, 13.0808544, -14.8701668, 14.9166603
23: -14.5488243, 0.3075175, -14.5698719, 0.3248906, -14.2565994, 14.2781296
24: -19.9302444, -5.1212649, -19.9287453, -5.1114388, -9.2559891, 9.2515297
25: -5.4405508, 10.8581371, -5.4367790, 10.8507967, -13.7508163, 13.7758331
26: -20.9626560, 1.2016733, -20.9604874, 1.1618118, -19.2450371, 19.2866440
27: -16.0065136, 2.1712573, -16.0122967, 2.1751952, -13.1505241, 13.2069740
28: -12.7627888, 4.6078753, -12.7870092, 4.6212225, -17.3840103, 17.3948841
29: -5.5370660, 11.8762894, -5.5513253, 11.8454876, -14.8527031, 14.9128952
30: -10.0345497, 6.2147799, -10.0340528, 6.1999321, -13.5212059, 13.5361633
31: -10.9532690, 6.9515300, -10.9637642, 6.9526143, -14.6034164, 14.6302910
32: -24.8940563, -4.5810928, -24.9240570, -4.5724440, -13.2443542, 13.2919464
33: -69.2869644, -40.1347809, -69.3153381, -40.1197395, -16.5948944, 16.6195679
34: -53.7322922, -30.9245987, -53.7679176, -30.9163914, -14.0854378, 14.1483650
35: -47.7996521, -26.0767746, -47.8214264, -26.0615673, -12.9842949, 12.9983635
36: -42.7961273, -19.2818336, -42.8065834, -19.2806377, -15.0615044, 15.0821838
37: -86.6604614, -55.5605927, -86.6691589, -55.5571785, -18.8869858, 18.8969460
38: -52.8981895, -24.3503685, -52.9349518, -24.3206272, -18.2934761, 18.3149109
39: -76.5212631, -44.6500168, -76.5478287, -44.6371613, -16.0336151, 16.0533218
40: -67.2451630, -43.5365524, -67.2471924, -43.5437508, -14.2452774, 14.3246441
41: -55.4188042, -32.9554939, -55.4477768, -32.9584389, -16.6180191, 16.7047005
42: -29.4514885, -9.8976173, -29.4700775, -9.8995247, -17.1780472, 17.2650681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4992264
time: 7.73 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665920, upper bound: 12.5344273
time: 7.99 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1136026, 3.6622713, -12.1130743, 3.6678247, -13.8573723, 13.8626137
1: -3.6551392, 7.3933382, -3.6502724, 7.3948364, -8.4674377, 8.4859753
2: -0.7523601, 13.4312477, -0.7363307, 13.4309025, -13.4395599, 13.4330101
3: -1.1262094, 11.2967224, -1.1199867, 11.2984247, -12.0142136, 11.9986477
4: -11.0882864, 5.4644246, -11.0893440, 5.4832544, -14.6338882, 14.6317062
5: 1.8430157, 17.7425041, 1.8505573, 17.7420845, -15.8990688, 15.8919468
6: -39.8649139, -18.2879047, -39.8751984, -18.2510262, -15.1091423, 15.0831223
7: -3.5759923, 12.2470703, -3.5586510, 12.2544031, -13.6026535, 13.5945282
8: -6.7146816, 8.5646152, -6.7011843, 8.5672226, -12.1042824, 12.0797501
9: -4.7492027, 11.7058897, -4.7687683, 11.6943951, -12.9586525, 12.9800491
10: 1.3400664, 25.7578392, 1.3430467, 25.7278996, -20.8706970, 20.8944321
11: -11.4900551, 4.2912016, -11.4935102, 4.2853489, -15.7754040, 15.7847118
12: -11.8889446, 9.8450384, -11.8897018, 9.8227978, -14.9913101, 15.0076180
13: -18.5408363, 6.7143836, -18.5425453, 6.7070293, -16.5852432, 16.5617867
14: 4.9822960, 36.4336853, 4.9795055, 36.3853683, -26.6597748, 26.7167358
15: -8.6483393, 9.2253132, -8.6779480, 9.2471437, -17.8954830, 17.9032612
16: -16.7226219, 2.5582280, -16.7093506, 2.5443213, -14.7729416, 14.8030739
17: 6.2188382, 30.6574249, 6.2131648, 30.6357994, -17.2063599, 17.1917496
18: -14.3596916, 5.1140642, -14.3699055, 5.1216803, -14.3652554, 14.3768749
19: -20.2614708, -4.3361559, -20.2653255, -4.3308482, -14.5150299, 14.5129738
20: -2.4106433, 11.2059736, -2.4083838, 11.2092037, -12.5965385, 12.5920601
21: -11.0617743, 3.2590599, -11.0635653, 3.2492616, -14.3110361, 14.3226252
22: -3.6943736, 13.1023722, -3.6978924, 13.0862522, -14.9156036, 14.9154320
23: -14.5563574, 0.3040564, -14.5770903, 0.3194413, -14.2651520, 14.2808762
24: -19.9284935, -5.1211147, -19.9276867, -5.1176834, -9.2635155, 9.2552299
25: -5.4494038, 10.8591213, -5.4483147, 10.8452187, -13.7839394, 13.7761154
26: -20.9915581, 1.2026744, -21.0091133, 1.1809037, -19.2983856, 19.2902069
27: -16.0159988, 2.1647086, -15.9942923, 2.1627231, -13.1815414, 13.1854095
28: -12.7769518, 4.6027117, -12.7882223, 4.6134319, -17.3903847, 17.3909340
29: -5.5649405, 11.8767433, -5.5932608, 11.8614197, -14.9046097, 14.9110870
30: -10.0432835, 6.2218180, -10.0426340, 6.2008619, -13.5339241, 13.5437508
31: -10.9584408, 6.9489202, -10.9572153, 6.9488368, -14.6224976, 14.6228218
32: -24.9033527, -4.6017284, -24.8906498, -4.5927663, -13.2768288, 13.2414627
33: -69.3178253, -40.1286850, -69.2868042, -40.1134262, -16.6269188, 16.6029625
34: -53.7595787, -30.9277573, -53.7325821, -30.9213829, -14.1387100, 14.0836945
35: -47.8177795, -26.0797539, -47.8029938, -26.0717163, -13.0074158, 12.9807091
36: -42.8292236, -19.2912636, -42.8019753, -19.2910309, -15.0948944, 15.0546303
37: -86.6735764, -55.5553093, -86.6635284, -55.5516205, -18.9116936, 18.9031944
38: -52.9320602, -24.3642616, -52.9094925, -24.3447552, -18.3220139, 18.2876663
39: -76.5363770, -44.6433105, -76.5347748, -44.6277161, -16.0538826, 16.0421028
40: -67.2662811, -43.5281448, -67.2243958, -43.5307388, -14.2827721, 14.2800694
41: -55.4416618, -32.9629974, -55.4051895, -32.9711838, -16.6638336, 16.6455917
42: -29.4564438, -9.9154644, -29.4515114, -9.9124718, -17.2182732, 17.2301445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665193, upper bound: 12.4795875
time: 11.38 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665193, upper bound: 12.5147130
time: 28.82 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.1171236, 3.6580257, -12.1162844, 3.6663327, -13.8581390, 13.8607750
1: -3.6644945, 7.3914900, -3.6578579, 7.3962507, -8.4724960, 8.4777946
2: -0.7547544, 13.4315119, -0.7378840, 13.4323978, -13.4456406, 13.4330673
3: -1.1270883, 11.3064766, -1.1218516, 11.3081112, -12.0121918, 12.0079193
4: -11.0952291, 5.4659472, -11.0941935, 5.4831314, -14.6458893, 14.6387558
5: 1.8425980, 17.7471809, 1.8518186, 17.7479439, -15.9053459, 15.8953629
6: -39.8591347, -18.2660828, -39.8863068, -18.2280083, -15.0982361, 15.1099663
7: -3.5761087, 12.2463837, -3.5606396, 12.2575264, -13.6008072, 13.5893517
8: -6.7182832, 8.5650463, -6.7042365, 8.5681877, -12.1169281, 12.0802612
9: -4.7546873, 11.6864309, -4.7761049, 11.6973143, -12.9685631, 12.9685936
10: 1.3311558, 25.7396774, 1.3268890, 25.7380047, -20.8919830, 20.8848648
11: -11.4905720, 4.2859020, -11.4951334, 4.2852564, -15.7758284, 15.7810354
12: -11.8923321, 9.8413172, -11.8992186, 9.8271856, -15.0009499, 15.0077133
13: -18.5467415, 6.7064219, -18.5503483, 6.7113929, -16.6073341, 16.5527687
14: 4.9699020, 36.3967972, 4.9587107, 36.4001045, -26.6933441, 26.6866989
15: -8.6575975, 9.2245054, -8.6882372, 9.2509432, -17.9085407, 17.9127426
16: -16.7294159, 2.5461874, -16.7196312, 2.5415707, -14.7718048, 14.7953491
17: 6.2148724, 30.6516781, 6.1985474, 30.6439590, -17.2006760, 17.2044144
18: -14.3660793, 5.1155548, -14.3734055, 5.1242199, -14.3728180, 14.3750019
19: -20.2638798, -4.3335400, -20.2699547, -4.3268061, -14.5171356, 14.5224533
20: -2.4087882, 11.2166328, -2.4142911, 11.2231445, -12.6041565, 12.6050186
21: -11.0649128, 3.2597532, -11.0673866, 3.2503152, -14.3152275, 14.3271399
22: -3.6896033, 13.1039934, -3.6987703, 13.0886698, -14.9082298, 14.9208069
23: -14.5554018, 0.3078337, -14.5781374, 0.3235166, -14.2709732, 14.2880402
24: -19.9318371, -5.1205821, -19.9308643, -5.1171875, -9.2652817, 9.2567062
25: -5.4524145, 10.8586283, -5.4543781, 10.8473568, -13.7794609, 13.7819633
26: -20.9978085, 1.2025397, -21.0152302, 1.1834624, -19.3017464, 19.2955551
27: -16.0079002, 2.1718755, -16.0038090, 2.1745553, -13.1733131, 13.2023277
28: -12.7710972, 4.6074600, -12.7942104, 4.6215954, -17.3926926, 17.4016705
29: -5.5642605, 11.8769493, -5.5947404, 11.8631878, -14.8996353, 14.9217186
30: -10.0440474, 6.2163200, -10.0460825, 6.2028313, -13.5331192, 13.5395966
31: -10.9554062, 6.9511604, -10.9600983, 6.9505296, -14.6227303, 14.6272240
32: -24.8951931, -4.5766172, -24.9046631, -4.5667629, -13.2652626, 13.2704506
33: -69.2879333, -40.1257744, -69.2935944, -40.1098022, -16.5943336, 16.6020088
34: -53.7331161, -30.9190559, -53.7425041, -30.9122143, -14.1094551, 14.1061554
35: -47.8009186, -26.0772705, -47.8068771, -26.0687790, -12.9926262, 12.9883575
36: -42.8091888, -19.2823372, -42.8086853, -19.2798309, -15.0821609, 15.0714493
37: -86.6647568, -55.5537720, -86.6665421, -55.5488052, -18.9133224, 18.9030304
38: -52.9066200, -24.3515892, -52.9228516, -24.3288441, -18.3028641, 18.3168259
39: -76.5221863, -44.6430893, -76.5359879, -44.6284180, -16.0465851, 16.0431595
40: -67.2466354, -43.5227890, -67.2316360, -43.5238037, -14.2753258, 14.2955494
41: -55.4194336, -32.9533691, -55.4169884, -32.9586563, -16.6476898, 16.6725655
42: -29.4519749, -9.8901081, -29.4612942, -9.8886681, -17.2122726, 17.2545891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5323996, upper bound: 12.5092856
time: 6.56 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5676281, upper bound: 12.5092856
time: 11.28 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -12.1155863, 3.6582487, -12.1149940, 3.6717954, -13.8630447, 13.8607216
1: -3.6612470, 7.3917980, -3.6534429, 7.3998680, -8.4840927, 8.4799824
2: -0.7534424, 13.4317541, -0.7368270, 13.4343872, -13.4476013, 13.4337006
3: -1.1270639, 11.3033543, -1.1219438, 11.3057308, -12.0139923, 12.0138912
4: -11.0914869, 5.4662414, -11.0892878, 5.4841342, -14.6458893, 14.6365662
5: 1.8426213, 17.7460709, 1.8513575, 17.7478828, -15.9052620, 15.8947134
6: -39.8592529, -18.2709007, -39.8935776, -18.2344685, -15.0977173, 15.1142616
7: -3.5755172, 12.2464361, -3.5621443, 12.2607889, -13.6103592, 13.5902100
8: -6.7165456, 8.5652237, -6.7037849, 8.5692501, -12.1186104, 12.0810852
9: -4.7567320, 11.6863880, -4.7838917, 11.7175865, -12.9825745, 12.9730110
10: 1.3268919, 25.7395363, 1.3128557, 25.7572079, -20.9116898, 20.8965836
11: -11.4922485, 4.2852802, -11.5024776, 4.2912254, -15.7834740, 15.7877579
12: -11.8925333, 9.8410254, -11.9049015, 9.8321686, -15.0048294, 15.0127106
13: -18.5471230, 6.7063270, -18.5529861, 6.7220097, -16.6112823, 16.5547028
14: 4.9647846, 36.3968620, 4.9403238, 36.4381104, -26.7288742, 26.7019119
15: -8.6548824, 9.2249956, -8.6854591, 9.2552366, -17.9101181, 17.9104538
16: -16.7310333, 2.5462759, -16.7301521, 2.5544264, -14.7906837, 14.8026581
17: 6.2139735, 30.6516991, 6.1901393, 30.6506195, -17.1986275, 17.2180481
18: -14.3622761, 5.1158590, -14.3696775, 5.1258831, -14.3751965, 14.3779221
19: -20.2625923, -4.3332777, -20.2715302, -4.3256545, -14.5168877, 14.5280724
20: -2.4089808, 11.2154694, -2.4188402, 11.2220802, -12.6038666, 12.6064186
21: -11.0631218, 3.2596638, -11.0694199, 3.2509670, -14.3140888, 14.3290834
22: -3.6894498, 13.1044712, -3.7053103, 13.0929041, -14.9116745, 14.9329185
23: -14.5558395, 0.3078170, -14.5826159, 0.3248253, -14.2730598, 14.2905273
24: -19.9302902, -5.1206427, -19.9295387, -5.1168294, -9.2667007, 9.2582550
25: -5.4520578, 10.8586683, -5.4566283, 10.8488598, -13.7788734, 13.7893524
26: -20.9946823, 1.2026334, -21.0142441, 1.1855888, -19.3025894, 19.2993011
27: -16.0077667, 2.1729305, -16.0136757, 2.1780376, -13.1771851, 13.2038231
28: -12.7712498, 4.6082859, -12.8025208, 4.6249614, -17.3962116, 17.4108067
29: -5.5643139, 11.8766441, -5.5970774, 11.8648071, -14.9005127, 14.9290314
30: -10.0449610, 6.2160354, -10.0512676, 6.2093029, -13.5411530, 13.5442429
31: -10.9552097, 6.9517317, -10.9677753, 6.9525957, -14.6242027, 14.6338348
32: -24.8952599, -4.5831394, -24.9146175, -4.5755615, -13.2650452, 13.2803574
33: -69.2875824, -40.1213303, -69.3252258, -40.0964661, -16.6030121, 16.6354218
34: -53.7331314, -30.9159431, -53.7699165, -30.9012260, -14.1180305, 14.1326981
35: -47.8008041, -26.0754452, -47.8244476, -26.0612240, -12.9995689, 13.0079308
36: -42.8090630, -19.2809029, -42.8292007, -19.2736320, -15.0859642, 15.0879326
37: -86.6644897, -55.5521622, -86.6772308, -55.5433273, -18.9176254, 18.9103050
38: -52.9065552, -24.3496151, -52.9494019, -24.3211288, -18.3076248, 18.3320923
39: -76.5219727, -44.6407547, -76.5525360, -44.6215706, -16.0519714, 16.0593452
40: -67.2465057, -43.5198822, -67.2526855, -43.5167770, -14.2798157, 14.2940083
41: -55.4194603, -32.9507790, -55.4408264, -32.9504395, -16.6507149, 16.6846695
42: -29.4521713, -9.8982468, -29.4675026, -9.9007759, -17.2102661, 17.2586098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5323996, upper bound: 12.5510406
time: 9.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5676281, upper bound: 12.5510406
time: 9.03 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -12.1348381, 3.6721611, -12.1079254, 3.6597934, -13.8857384, 13.8624420
1: -3.6715117, 7.3914089, -3.6558785, 7.3782387, -8.4726868, 8.4819584
2: -0.7642527, 13.4238291, -0.7303644, 13.4110451, -13.4451141, 13.4239845
3: -1.1399482, 11.2969732, -1.1285565, 11.2864094, -12.0281868, 11.9970055
4: -11.1176777, 5.4751439, -11.0977793, 5.4642220, -14.6463776, 14.6492157
5: 1.8263636, 17.7401295, 1.8442917, 17.7318764, -15.9055128, 15.8940239
6: -39.9372215, -18.2444172, -39.9211540, -18.2472782, -15.0947762, 15.1728249
7: -3.6104448, 12.2476358, -3.5748842, 12.2300014, -13.6069641, 13.5881920
8: -6.7188334, 8.5647879, -6.6951723, 8.5570087, -12.0990028, 12.0941906
9: -4.7806416, 11.7360077, -4.7724957, 11.7030067, -12.9944267, 12.9825745
10: 1.3126388, 25.7598362, 1.3308997, 25.7182484, -20.8892746, 20.9164658
11: -11.5030699, 4.2927127, -11.4954901, 4.2844062, -15.7874756, 15.7882023
12: -11.8892164, 9.8608694, -11.8679924, 9.8138256, -14.9731865, 15.0194397
13: -18.5413685, 6.7425756, -18.5376701, 6.7252808, -16.6197815, 16.5368919
14: 4.9701719, 36.4697800, 5.0117121, 36.4098015, -26.7118378, 26.6823120
15: -8.6882172, 9.2918072, -8.6883917, 9.3038149, -17.9920311, 17.9801979
16: -16.7480373, 2.5363600, -16.7205849, 2.5096192, -14.7746544, 14.8078346
17: 6.2259731, 30.6827717, 6.2655106, 30.6394024, -17.2061386, 17.1745377
18: -14.3935928, 5.1252594, -14.3848486, 5.1181879, -14.3963966, 14.4045525
19: -20.2730141, -4.3239732, -20.2605553, -4.3261423, -14.5274353, 14.5194473
20: -2.4196558, 11.2166567, -2.4012673, 11.2057810, -12.5949478, 12.5985146
21: -11.0726738, 3.2633476, -11.0582409, 3.2424097, -14.3150835, 14.3215885
22: -3.6878648, 13.1431112, -3.6639738, 13.0974636, -14.9177856, 14.9329033
23: -14.5814095, 0.3466010, -14.5692844, 0.3445015, -14.3038635, 14.2936401
24: -19.9334450, -5.1118231, -19.9286842, -5.1072512, -9.2557335, 9.2664070
25: -5.4507618, 10.8867788, -5.4313231, 10.8618870, -13.7834396, 13.7858276
26: -20.9846764, 1.2552531, -20.9584999, 1.1879225, -19.2880020, 19.2961426
27: -16.0319939, 2.1748683, -15.9986162, 2.1663027, -13.1897392, 13.2040253
28: -12.7999735, 4.6443610, -12.7771616, 4.6341729, -17.4341469, 17.4215221
29: -5.5729251, 11.9247208, -5.5498767, 11.8696995, -14.9188347, 14.9260406
30: -10.0419931, 6.2272615, -10.0294762, 6.1960955, -13.5339546, 13.5493546
31: -10.9881973, 6.9501691, -10.9694204, 6.9495516, -14.6339417, 14.6349411
32: -24.9361649, -4.5823874, -24.9192295, -4.5890021, -13.2570763, 13.2849541
33: -69.3504791, -40.1088333, -69.2961578, -40.1308899, -16.6306801, 16.6382065
34: -53.7926254, -30.9142456, -53.7508507, -30.9334602, -14.1024857, 14.1404305
35: -47.8250008, -26.0701828, -47.8046417, -26.0695972, -12.9901276, 12.9879761
36: -42.8160477, -19.2867565, -42.7833595, -19.2964668, -15.0794678, 15.0596924
37: -86.6778412, -55.5521393, -86.6605835, -55.5594826, -18.8879852, 18.9172249
38: -52.9695282, -24.3323345, -52.9204483, -24.3418407, -18.3220367, 18.3288422
39: -76.5640335, -44.6307335, -76.5453949, -44.6397171, -16.0676651, 16.0784988
40: -67.3043823, -43.5373077, -67.2381210, -43.5573044, -14.2866707, 14.3447704
41: -55.4644089, -32.9606056, -55.4241638, -32.9782867, -16.6400909, 16.6839981
42: -29.4695549, -9.9123726, -29.4608555, -9.9101601, -17.2030258, 17.2478485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5302547, upper bound: 12.5133523
time: 6.61 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5654599, upper bound: 12.5133523
time: 6.59 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.1383696, 3.6678972, -12.1111469, 3.6583223, -13.8864937, 13.8605804
1: -3.6808767, 7.3895674, -3.6634545, 7.3796453, -8.4777184, 8.4737854
2: -0.7665980, 13.4240847, -0.7319062, 13.4125423, -13.4511948, 13.4240799
3: -1.1408374, 11.3067198, -1.1304249, 11.2960854, -12.0261841, 12.0062485
4: -11.1246128, 5.4766364, -11.1026287, 5.4640932, -14.6583481, 14.6562157
5: 1.8259435, 17.7448120, 1.8455386, 17.7377415, -15.9117985, 15.8992729
6: -39.9314423, -18.2225742, -39.9322662, -18.2242641, -15.0839272, 15.1996460
7: -3.6105301, 12.2469387, -3.5769076, 12.2331772, -13.6051407, 13.5829773
8: -6.7224417, 8.5652256, -6.6982083, 8.5580215, -12.1116714, 12.0947189
9: -4.7860966, 11.7165365, -4.7798557, 11.7059431, -13.0043564, 12.9711494
10: 1.3037419, 25.7416382, 1.3147454, 25.7283707, -20.9105225, 20.9068298
11: -11.5035639, 4.2874217, -11.4971247, 4.2843113, -15.7878752, 15.7845459
12: -11.8926086, 9.8571472, -11.8775015, 9.8182049, -14.9828262, 15.0195465
13: -18.5471916, 6.7346392, -18.5454731, 6.7297096, -16.6418457, 16.5279388
14: 4.9577827, 36.4329758, 4.9909124, 36.4245644, -26.7453461, 26.6523514
15: -8.6974831, 9.2910032, -8.6986923, 9.3076057, -18.0050888, 17.9896965
16: -16.7548599, 2.5243526, -16.7309151, 2.5068893, -14.7734680, 14.8001595
17: 6.2220278, 30.6770515, 6.2508979, 30.6475639, -17.2004395, 17.1872253
18: -14.3999939, 5.1267281, -14.3883724, 5.1207123, -14.4039688, 14.4026909
19: -20.2754326, -4.3213634, -20.2652035, -4.3221149, -14.5295334, 14.5289192
20: -2.4178009, 11.2272844, -2.4071882, 11.2197008, -12.6026001, 12.6114540
21: -11.0757866, 3.2640224, -11.0620632, 3.2434654, -14.3192520, 14.3260860
22: -3.6830814, 13.1447096, -3.6648498, 13.0999088, -14.9103775, 14.9382553
23: -14.5804596, 0.3503540, -14.5703201, 0.3485909, -14.3096962, 14.3007965
24: -19.9367695, -5.1113076, -19.9318542, -5.1067643, -9.2574883, 9.2678871
25: -5.4537745, 10.8862743, -5.4373717, 10.8640308, -13.7789764, 13.7916794
26: -20.9909668, 1.2551537, -20.9646263, 1.1904833, -19.2913437, 19.3014984
27: -16.0238724, 2.1820273, -16.0080986, 2.1781261, -13.1815300, 13.2209663
28: -12.7941256, 4.6490993, -12.7831640, 4.6423526, -17.4364777, 17.4322624
29: -5.5722613, 11.9249315, -5.5513940, 11.8714790, -14.9138184, 14.9366798
30: -10.0427742, 6.2217579, -10.0329437, 6.1980805, -13.5331459, 13.5451965
31: -10.9851913, 6.9523830, -10.9722824, 6.9512148, -14.6341515, 14.6393433
32: -24.9280052, -4.5573025, -24.9332695, -4.5629816, -13.2454796, 13.3139343
33: -69.3205719, -40.1058922, -69.3029327, -40.1272964, -16.5981522, 16.6372337
34: -53.7661972, -30.9055805, -53.7607803, -30.9242878, -14.0732155, 14.1628876
35: -47.8081055, -26.0676804, -47.8085251, -26.0666466, -12.9753342, 12.9956512
36: -42.7959976, -19.2778625, -42.7900963, -19.2852478, -15.0667191, 15.0765152
37: -86.6689835, -55.5505524, -86.6636658, -55.5566406, -18.8896065, 18.9171181
38: -52.9440842, -24.3196850, -52.9338379, -24.3259735, -18.3029022, 18.3580017
39: -76.5498810, -44.6304932, -76.5466385, -44.6404533, -16.0603905, 16.0795441
40: -67.2847366, -43.5319519, -67.2453613, -43.5503693, -14.2792587, 14.3602448
41: -55.4422264, -32.9509888, -55.4358826, -32.9658051, -16.6239243, 16.7109795
42: -29.4650536, -9.8870296, -29.4706039, -9.8863726, -17.1970215, 17.2723389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5314954, upper bound: 12.5093835
time: 7.37 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5667201, upper bound: 12.5093835
time: 21.43 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -12.1368313, 3.6681430, -12.1098490, 3.6637611, -13.8914070, 13.8605385
1: -3.6776371, 7.3898602, -3.6590438, 7.3832464, -8.4893227, 8.4759541
2: -0.7653412, 13.4243116, -0.7308431, 13.4145269, -13.4531784, 13.4246979
3: -1.1408057, 11.3036175, -1.1305237, 11.2937298, -12.0279846, 12.0122242
4: -11.1208935, 5.4769092, -11.0977192, 5.4650803, -14.6583405, 14.6540451
5: 1.8259892, 17.7437057, 1.8450899, 17.7376747, -15.9116859, 15.8986158
6: -39.9315872, -18.2273865, -39.9395103, -18.2307415, -15.0834007, 15.2039070
7: -3.6099548, 12.2469578, -3.5783782, 12.2364340, -13.6146698, 13.5838432
8: -6.7206826, 8.5653725, -6.6977625, 8.5590658, -12.1133652, 12.0955448
9: -4.7881103, 11.7164917, -4.7876501, 11.7261963, -13.0183601, 12.9755440
10: 1.2994905, 25.7415619, 1.3007154, 25.7475662, -20.9302368, 20.9185333
11: -11.5052414, 4.2868080, -11.5044556, 4.2902646, -15.7955055, 15.7912636
12: -11.8928347, 9.8568487, -11.8832178, 9.8231916, -14.9867134, 15.0245285
13: -18.5476074, 6.7345905, -18.5480957, 6.7403216, -16.6458015, 16.5298920
14: 4.9527006, 36.4330292, 4.9725485, 36.4625473, -26.7809143, 26.6675491
15: -8.6947527, 9.2915192, -8.6959248, 9.3119068, -18.0066605, 17.9874439
16: -16.7564926, 2.5244141, -16.7414379, 2.5197477, -14.7923203, 14.8073959
17: 6.2211175, 30.6770630, 6.2424955, 30.6542072, -17.1983833, 17.2008667
18: -14.3961716, 5.1270342, -14.3846569, 5.1224065, -14.4063568, 14.4056129
19: -20.2741470, -4.3211226, -20.2667656, -4.3209348, -14.5292740, 14.5345535
20: -2.4180002, 11.2261391, -2.4117472, 11.2186508, -12.6022949, 12.6128578
21: -11.0740204, 3.2639446, -11.0641308, 3.2441087, -14.3181286, 14.3280754
22: -3.6829638, 13.1452084, -3.6713881, 13.1041279, -14.9138336, 14.9503860
23: -14.5808487, 0.3503675, -14.5748014, 0.3498793, -14.3117485, 14.3033142
24: -19.9352264, -5.1113534, -19.9305286, -5.1064105, -9.2589264, 9.2694435
25: -5.4533930, 10.8863010, -5.4396524, 10.8655090, -13.7783890, 13.7990532
26: -20.9877796, 1.2552490, -20.9636421, 1.1925645, -19.2921753, 19.3052597
27: -16.0237503, 2.1830788, -16.0179920, 2.1816308, -13.1853600, 13.2224655
28: -12.7942781, 4.6499104, -12.7914829, 4.6456914, -17.4399700, 17.4413929
29: -5.5723267, 11.9246273, -5.5536861, 11.8731079, -14.9146957, 14.9440193
30: -10.0436764, 6.2214956, -10.0381212, 6.2045469, -13.5411720, 13.5498543
31: -10.9849939, 6.9529624, -10.9799681, 6.9533176, -14.6356392, 14.6459503
32: -24.9280910, -4.5637941, -24.9432030, -4.5717993, -13.2452812, 13.3238373
33: -69.3202667, -40.1014824, -69.3345947, -40.1139908, -16.6068230, 16.6706467
34: -53.7661972, -30.9024754, -53.7881546, -30.9132996, -14.0817642, 14.1894341
35: -47.8080444, -26.0658913, -47.8260803, -26.0591278, -12.9822884, 13.0152054
36: -42.7958794, -19.2764359, -42.8105354, -19.2790680, -15.0705719, 15.0930214
37: -86.6687317, -55.5488815, -86.6743088, -55.5511780, -18.8938980, 18.9243546
38: -52.9439774, -24.3177071, -52.9603500, -24.3182182, -18.3076973, 18.3732376
39: -76.5496674, -44.6281357, -76.5631180, -44.6336517, -16.0657578, 16.0957184
40: -67.2845764, -43.5290527, -67.2663803, -43.5433350, -14.2837334, 14.3586903
41: -55.4422607, -32.9484215, -55.4597092, -32.9575424, -16.6269379, 16.7230453
42: -29.4652710, -9.8951626, -29.4768105, -9.8984690, -17.1950264, 17.2763214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5314954, upper bound: 12.5508793
time: 11.21 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5667201, upper bound: 12.5508793
time: 20.65 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1366014, 3.6850286, -12.1184006, 3.6809506, -13.8939972, 13.8875809
1: -3.6718116, 7.4035988, -3.6601164, 7.3981848, -8.4778290, 8.5020313
2: -0.7654744, 13.4379654, -0.7427936, 13.4348841, -13.4580612, 13.4504623
3: -1.1407350, 11.3069229, -1.1282327, 11.3033648, -12.0420914, 12.0063210
4: -11.1183577, 5.4889627, -11.1065435, 5.4876032, -14.6586838, 14.6717834
5: 1.8250227, 17.7474251, 1.8406420, 17.7444592, -15.9194365, 15.9067831
6: -39.9389992, -18.2459660, -39.9178352, -18.2506905, -15.1138153, 15.1683464
7: -3.6110961, 12.2637691, -3.5774403, 12.2569370, -13.6168213, 13.6233368
8: -6.7195444, 8.5741215, -6.7021499, 8.5723019, -12.1167183, 12.0899277
9: -4.7818308, 11.7424374, -4.7750340, 11.7144566, -13.0103378, 12.9973221
10: 1.3105955, 25.7677898, 1.3275790, 25.7327709, -20.9056549, 20.9454193
11: -11.5063152, 4.2934189, -11.5004730, 4.2869687, -15.7932835, 15.7938919
12: -11.9081573, 9.8621788, -11.8997202, 9.8247623, -15.0032539, 15.0353432
13: -18.5453815, 6.7436795, -18.5439072, 6.7215972, -16.5959587, 16.5909271
14: 4.9493752, 36.4703178, 4.9759064, 36.4056396, -26.7134247, 26.7308044
15: -8.6844311, 9.2926617, -8.6805382, 9.2854958, -17.9699268, 17.9731998
16: -16.7493782, 2.5581837, -16.7219696, 2.5455115, -14.8095398, 14.8152580
17: 6.1937809, 30.6833839, 6.2110519, 30.6495533, -17.2464104, 17.2186737
18: -14.3969765, 5.1301794, -14.3909712, 5.1267238, -14.4068222, 14.4163094
19: -20.2803535, -4.3237877, -20.2734108, -4.3235731, -14.5438843, 14.5334396
20: -2.4301658, 11.2178507, -2.4188018, 11.2121611, -12.6110268, 12.6103554
21: -11.0821400, 3.2644258, -11.0745869, 3.2501674, -14.3323078, 14.3390121
22: -3.7103186, 13.1436253, -3.7008803, 13.1095352, -14.9592896, 14.9491844
23: -14.5883999, 0.3469248, -14.5820475, 0.3444054, -14.3203278, 14.3060684
24: -19.9334831, -5.1111746, -19.9294395, -5.1126270, -9.2664642, 9.2731514
25: -5.4622593, 10.8872967, -5.4511833, 10.8599367, -13.8115768, 13.7993164
26: -21.0167103, 1.2562509, -21.0122147, 1.2116616, -19.3455276, 19.3087769
27: -16.0332680, 2.1765265, -16.0000114, 2.1691344, -13.2164001, 13.2008781
28: -12.8084202, 4.6447496, -12.7927170, 4.6379151, -17.4463348, 17.4374657
29: -5.6002183, 11.9250708, -5.5956478, 11.8890419, -14.9666252, 14.9421959
30: -10.0524330, 6.2285261, -10.0467033, 6.2054873, -13.5538864, 13.5574570
31: -10.9901466, 6.9503713, -10.9734306, 6.9495392, -14.6547241, 14.6384964
32: -24.9373550, -4.5844030, -24.9097576, -4.5921392, -13.2777557, 13.2733688
33: -69.3511200, -40.0953979, -69.3060303, -40.1076584, -16.6388130, 16.6540565
34: -53.7934875, -30.9056053, -53.7528648, -30.9182682, -14.1350670, 14.1247711
35: -47.8261948, -26.0688972, -47.8076477, -26.0692234, -13.0054131, 12.9975471
36: -42.8289452, -19.2858562, -42.8060150, -19.2894516, -15.1039391, 15.0654526
37: -86.6818466, -55.5436821, -86.6687775, -55.5456085, -18.9186783, 18.9306030
38: -52.9779053, -24.3316231, -52.9349442, -24.3423290, -18.3361664, 18.3460083
39: -76.5647049, -44.6214523, -76.5501099, -44.6241684, -16.0859985, 16.0844803
40: -67.3056793, -43.5206604, -67.2435760, -43.5302849, -14.3212013, 14.3141251
41: -55.4650650, -32.9559402, -55.4171982, -32.9702721, -16.6727753, 16.6639481
42: -29.4702415, -9.9130020, -29.4583073, -9.9114227, -17.2351990, 17.2414322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5314252, upper bound: 12.5311252
time: 11.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5666467, upper bound: 12.5311252
time: 6.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.1401548, 3.6807771, -12.1216221, 3.6794615, -13.8947411, 13.8857117
1: -3.6811697, 7.4017658, -3.6677032, 7.3995762, -8.4828796, 8.4938622
2: -0.7678450, 13.4382401, -0.7443563, 13.4364223, -13.4641342, 13.4505196
3: -1.1416237, 11.3166971, -1.1301063, 11.3130274, -12.0400772, 12.0155754
4: -11.1252937, 5.4904346, -11.1113815, 5.4874582, -14.6706543, 14.6787796
5: 1.8246241, 17.7521381, 1.8419056, 17.7503300, -15.9257059, 15.9102325
6: -39.9332085, -18.2240963, -39.9289474, -18.2276649, -15.1029320, 15.1951866
7: -3.6112018, 12.2630701, -3.5794249, 12.2600899, -13.6150131, 13.6181450
8: -6.7231579, 8.5745497, -6.7051897, 8.5732861, -12.1293526, 12.0904388
9: -4.7873163, 11.7230101, -4.7823606, 11.7173815, -13.0202484, 12.9859123
10: 1.3016715, 25.7496033, 1.3113866, 25.7428589, -20.9269028, 20.9357910
11: -11.5068054, 4.2881303, -11.5020981, 4.2868810, -15.7936859, 15.7902279
12: -11.9115763, 9.8584690, -11.9092407, 9.8291588, -15.0128860, 15.0354347
13: -18.5512810, 6.7357416, -18.5516796, 6.7260537, -16.6180115, 16.5819855
14: 4.9369802, 36.4334793, 4.9550562, 36.4203873, -26.7469406, 26.7007980
15: -8.6937180, 9.2918530, -8.6908169, 9.2892723, -17.9829903, 17.9826698
16: -16.7561874, 2.5461693, -16.7323112, 2.5427666, -14.8084602, 14.8076096
17: 6.1898766, 30.6776466, 6.1964641, 30.6576958, -17.2407227, 17.2313347
18: -14.4033546, 5.1316552, -14.3944435, 5.1292410, -14.4143906, 14.4144459
19: -20.2827816, -4.3211689, -20.2780609, -4.3195238, -14.5460052, 14.5429344
20: -2.4282930, 11.2284794, -2.4247286, 11.2261038, -12.6186905, 12.6233063
21: -11.0852413, 3.2651052, -11.0784073, 3.2512183, -14.3364601, 14.3435125
22: -3.7055399, 13.1452112, -3.7017348, 13.1119814, -14.9519157, 14.9545326
23: -14.5874405, 0.3506608, -14.5830479, 0.3485236, -14.3261795, 14.3132133
24: -19.9368401, -5.1106844, -19.9326229, -5.1121731, -9.2682343, 9.2746315
25: -5.4652586, 10.8868141, -5.4572368, 10.8620777, -13.8070946, 13.8051643
26: -21.0229912, 1.2561212, -21.0183372, 1.2142224, -19.3489075, 19.3141556
27: -16.0251579, 2.1837015, -16.0095100, 2.1809659, -13.2081909, 13.2178268
28: -12.8025751, 4.6494937, -12.7986698, 4.6460767, -17.4486523, 17.4481640
29: -5.5995274, 11.9252796, -5.5971460, 11.8908253, -14.9616318, 14.9528236
30: -10.0531998, 6.2229824, -10.0501432, 6.2074480, -13.5530701, 13.5532990
31: -10.9871407, 6.9526162, -10.9763193, 6.9512243, -14.6549454, 14.6428909
32: -24.9292049, -4.5593081, -24.9238491, -4.5661278, -13.2661858, 13.3023758
33: -69.3212051, -40.0924683, -69.3128357, -40.1040649, -16.6062469, 16.6530914
34: -53.7670288, -30.8969383, -53.7627831, -30.9091034, -14.1058083, 14.1472015
35: -47.8092957, -26.0664101, -47.8115387, -26.0662918, -12.9906082, 13.0052032
36: -42.8089333, -19.2769299, -42.8127060, -19.2782764, -15.0911980, 15.0822792
37: -86.6730042, -55.5420609, -86.6717529, -55.5427856, -18.9202614, 18.9304657
38: -52.9523849, -24.3188992, -52.9482803, -24.3264122, -18.3170624, 18.3751831
39: -76.5505219, -44.6211967, -76.5512695, -44.6248817, -16.0787354, 16.0855217
40: -67.2860641, -43.5153122, -67.2508087, -43.5233955, -14.3137856, 14.3295994
41: -55.4428368, -32.9463120, -55.4289589, -32.9577255, -16.6565933, 16.6909027
42: -29.4657650, -9.8876572, -29.4680462, -9.8876114, -17.2292557, 17.2658844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5325247, upper bound: 12.5260153
time: 11.53 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677528, upper bound: 12.5260153
time: 16.08 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -12.1386175, 3.6809936, -12.1202974, 3.6849055, -13.8996277, 13.8856926
1: -3.6779108, 7.4020462, -3.6632867, 7.4031949, -8.4944649, 8.4960194
2: -0.7665870, 13.4384613, -0.7432689, 13.4384012, -13.4660873, 13.4511642
3: -1.1415856, 11.3135910, -1.1302032, 11.3106718, -12.0418739, 12.0215473
4: -11.1215706, 5.4906931, -11.1064663, 5.4884582, -14.6706772, 14.6766319
5: 1.8246431, 17.7510319, 1.8414660, 17.7502594, -15.9256163, 15.9095659
6: -39.9333534, -18.2289619, -39.9362030, -18.2341232, -15.1024361, 15.1994438
7: -3.6106017, 12.2631130, -3.5809240, 12.2633190, -13.6245346, 13.6190186
8: -6.7213993, 8.5747252, -6.7047291, 8.5743818, -12.1310501, 12.0912552
9: -4.7893414, 11.7229710, -4.7901802, 11.7376223, -13.0342445, 12.9903183
10: 1.2974720, 25.7494583, 1.2973409, 25.7620735, -20.9466400, 20.9475403
11: -11.5084810, 4.2874947, -11.5094414, 4.2928381, -15.8013191, 15.7969360
12: -11.9117355, 9.8581810, -11.9149265, 9.8341656, -15.0167923, 15.0404282
13: -18.5516338, 6.7356315, -18.5543060, 6.7367263, -16.6219635, 16.5839157
14: 4.9318438, 36.4335022, 4.9366846, 36.4584198, -26.7825012, 26.7160110
15: -8.6909695, 9.2923431, -8.6880856, 9.2935610, -17.9845314, 17.9804287
16: -16.7578220, 2.5462127, -16.7428284, 2.5556440, -14.8273087, 14.8149147
17: 6.1889348, 30.6776695, 6.1880426, 30.6643658, -17.2386780, 17.2449913
18: -14.3995304, 5.1319523, -14.3907547, 5.1309552, -14.4167709, 14.4173737
19: -20.2814999, -4.3209362, -20.2796288, -4.3183727, -14.5457230, 14.5485535
20: -2.4284956, 11.2273426, -2.4292734, 11.2250366, -12.6183929, 12.6247139
21: -11.0834656, 3.2650354, -11.0804758, 3.2518487, -14.3353138, 14.3455114
22: -3.7054226, 13.1457100, -3.7083008, 13.1161747, -14.9553566, 14.9666481
23: -14.5878334, 0.3506784, -14.5875378, 0.3498008, -14.3282280, 14.3157272
24: -19.9352951, -5.1107063, -19.9312935, -5.1117859, -9.2696495, 9.2761879
25: -5.4649029, 10.8868408, -5.4594955, 10.8635559, -13.8064995, 13.8125458
26: -21.0198269, 1.2562385, -21.0173893, 1.2163262, -19.3497162, 19.3179092
27: -16.0250168, 2.1847365, -16.0193748, 2.1844728, -13.2120285, 13.2193298
28: -12.8027344, 4.6503091, -12.8070145, 4.6494346, -17.4521694, 17.4573231
29: -5.5995960, 11.9249573, -5.5994310, 11.8924446, -14.9625511, 14.9601593
30: -10.0541134, 6.2226868, -10.0553350, 6.2139082, -13.5610962, 13.5579300
31: -10.9869194, 6.9531727, -10.9840002, 6.9533138, -14.6564331, 14.6495018
32: -24.9293213, -4.5658216, -24.9337864, -4.5749159, -13.2659874, 13.3122787
33: -69.3209229, -40.0880699, -69.3444824, -40.0907745, -16.6149139, 16.6865387
34: -53.7670364, -30.8938255, -53.7901611, -30.8981209, -14.1143646, 14.1737633
35: -47.8092194, -26.0646133, -47.8291054, -26.0587349, -12.9975548, 13.0247650
36: -42.8088379, -19.2754898, -42.8331871, -19.2721195, -15.0950470, 15.0987778
37: -86.6727905, -55.5404663, -86.6824799, -55.5373154, -18.9246254, 18.9377136
38: -52.9523697, -24.3169823, -52.9747925, -24.3186684, -18.3218613, 18.3904495
39: -76.5503845, -44.6188965, -76.5677872, -44.6179886, -16.0840988, 16.1016922
40: -67.2858887, -43.5124245, -67.2718582, -43.5163422, -14.3182564, 14.3280487
41: -55.4428940, -32.9437714, -55.4527512, -32.9494934, -16.6596375, 16.7029915
42: -29.4659576, -9.8957520, -29.4742241, -9.8997326, -17.2272224, 17.2698822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 952

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5325247, upper bound: 12.5677523
time: 8.23 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677528, upper bound: 12.5677523
time: 12.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.70 seconds
IS_A1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4220940, upper bound: 12.5657701
IS_A1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4552077, upper bound: 12.5657701
IS_A1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4630811, upper bound: 12.5657701
IS_A1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4959760, upper bound: 12.5657701
IS_A1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4592958, upper bound: 12.5669280
IS_A1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.4931458, upper bound: 12.5669280
IS_A1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5006482, upper bound: 12.5669280
IS_A1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5340302, upper bound: 12.5669280
IS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5653244, upper bound: 12.4628685
IS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5653244, upper bound: 12.4974587
IS_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4574710
IS_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4927019
IS_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665920, upper bound: 12.4992264
IS_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665920, upper bound: 12.5344273
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665193, upper bound: 12.4795875
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5665193, upper bound: 12.5147130
IS_A2_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5323996, upper bound: 12.5092856
IS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5676281, upper bound: 12.5092856
IS_A2_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5323996, upper bound: 12.5510406
IS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5676281, upper bound: 12.5510406
IS_A2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5302547, upper bound: 12.5133523
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5654599, upper bound: 12.5133523
IS_A2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5314954, upper bound: 12.5093835
IS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5667201, upper bound: 12.5093835
IS_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5314954, upper bound: 12.5508793
IS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5667201, upper bound: 12.5508793
IS_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5314252, upper bound: 12.5311252
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5666467, upper bound: 12.5311252
IS_A2_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5325247, upper bound: 12.5260153
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5677528, upper bound: 12.5260153
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5325247, upper bound: 12.5677523
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.70
Output dim: 14, lower bound: -12.5677528, upper bound: 12.5677523

## BFS IS instance: IS_A1_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -12.1155491, 3.6707621, -12.1195211, 3.6755996, -13.8624611, 13.8737144
1: -3.6612267, 7.3892188, -3.6648736, 7.3956957, -8.4625092, 8.4840412
2: -0.7357342, 13.4209423, -0.7403869, 13.4308186, -13.4224625, 13.4298897
3: -1.1112341, 11.2904949, -1.1205777, 11.3040924, -11.9978867, 11.9832497
4: -11.1075411, 5.4740868, -11.1112690, 5.4812689, -14.6421509, 14.6584854
5: 1.8573546, 17.7309494, 1.8513374, 17.7425804, -15.8852253, 15.8796120
6: -39.8953934, -18.2496490, -39.9175987, -18.2354031, -15.0876923, 15.1510201
7: -3.5521560, 12.2391348, -3.5647740, 12.2504320, -13.5396805, 13.5868607
8: -6.6984749, 8.5621271, -6.7034235, 8.5690918, -12.0872879, 12.0727692
9: -4.7643838, 11.7044382, -4.7798209, 11.7120733, -12.9870834, 12.9742546
10: 1.3534913, 25.7220078, 1.3255796, 25.7365875, -20.8717346, 20.9072800
11: -11.4752893, 4.2857885, -11.4913950, 4.2868729, -15.7621622, 15.7771835
12: -11.8756104, 9.8208618, -11.8948708, 9.8273392, -14.9749603, 14.9787102
13: -18.5392113, 6.7179451, -18.5519485, 6.7250257, -16.5668793, 16.6138954
14: 5.0343399, 36.4039421, 4.9870329, 36.4198647, -26.6491089, 26.6689835
15: -8.6851549, 9.2563782, -8.6902285, 9.2738266, -17.9589806, 17.9466057
16: -16.7069893, 2.5304732, -16.7235622, 2.5345604, -14.7703552, 14.7935257
17: 6.2535300, 30.6467724, 6.2236342, 30.6564827, -17.1780968, 17.1937561
18: -14.3889513, 5.1091642, -14.3928432, 5.1204834, -14.3944321, 14.3935165
19: -20.2663860, -4.3280911, -20.2746258, -4.3232441, -14.5245018, 14.5246086
20: -2.4038632, 11.2111197, -2.4166222, 11.2182865, -12.5944557, 12.5969200
21: -11.0591831, 3.2485051, -11.0709200, 3.2499413, -14.3091240, 14.3194256
22: -3.6833313, 13.0949326, -3.6938069, 13.1033249, -14.9232483, 14.8876343
23: -14.5725555, 0.3375454, -14.5779800, 0.3441753, -14.3121300, 14.2885704
24: -19.9294319, -5.1267228, -19.9320755, -5.1200190, -9.2479897, 9.2494278
25: -5.4362860, 10.8475199, -5.4466147, 10.8555126, -13.7807426, 13.7587433
26: -20.9886417, 1.1906996, -21.0067387, 1.2020524, -19.3130035, 19.2433167
27: -15.9980812, 2.1571040, -16.0083637, 2.1702132, -13.2097626, 13.1890450
28: -12.7834949, 4.6320009, -12.7941093, 4.6404991, -17.4239941, 17.4261093
29: -5.5765352, 11.8806219, -5.5885568, 11.8858528, -14.9328575, 14.8925400
30: -10.0289135, 6.2034464, -10.0421467, 6.2065077, -13.5301247, 13.5289230
31: -10.9656792, 6.9447985, -10.9731817, 6.9494152, -14.6375389, 14.6291237
32: -24.8883228, -4.5897827, -24.9123001, -4.5772266, -13.2424431, 13.2590561
33: -69.3039322, -40.1327400, -69.3124084, -40.1129456, -16.6029816, 16.6222763
34: -53.7507973, -30.9410896, -53.7625046, -30.9187851, -14.1047859, 14.1011124
35: -47.8137932, -26.0814018, -47.8185196, -26.0711021, -12.9793472, 12.9811401
36: -42.8114929, -19.3030224, -42.8208084, -19.2853241, -15.0825119, 15.0548744
37: -86.6661224, -55.5818062, -86.6736450, -55.5608368, -18.8856812, 18.8767242
38: -52.9285736, -24.3551350, -52.9454575, -24.3364716, -18.3040924, 18.3364868
39: -76.5526962, -44.6381721, -76.5575638, -44.6286736, -16.0636253, 16.0613747
40: -67.2397919, -43.5574265, -67.2497253, -43.5387192, -14.3241119, 14.3006668
41: -55.4142456, -32.9897194, -55.4280930, -32.9706650, -16.6658401, 16.6441612
42: -29.4532089, -9.8979082, -29.4664822, -9.8877869, -17.2432671, 17.2218246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4014289, upper bound: 12.5631753
time: 12.68 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4211745, upper bound: 12.5655894
time: 14.77 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -12.1230679, 3.6740651, -12.1206923, 3.6760447, -13.8729439, 13.8771935
1: -3.6621046, 7.3913832, -3.6635752, 7.3959341, -8.4636002, 8.4887066
2: -0.7419192, 13.4277134, -0.7424394, 13.4310398, -13.4308128, 13.4426041
3: -1.1288835, 11.3133926, -1.1291448, 11.3049955, -12.0141106, 12.0148201
4: -11.1111383, 5.4801903, -11.1113892, 5.4825764, -14.6436157, 14.6647377
5: 1.8413148, 17.7513790, 1.8431683, 17.7434673, -15.8987961, 15.9082108
6: -39.9165115, -18.2236481, -39.9271584, -18.2348480, -15.1061211, 15.1933861
7: -3.5816524, 12.2697086, -3.5783186, 12.2513599, -13.5645752, 13.6310768
8: -6.7032890, 8.5660982, -6.7029123, 8.5692940, -12.0895615, 12.0918674
9: -4.7736835, 11.7151985, -4.7801261, 11.7161541, -13.0036011, 12.9834213
10: 1.3354349, 25.7321930, 1.3243585, 25.7414169, -20.8947525, 20.9165115
11: -11.4947376, 4.3063126, -11.4973574, 4.2869611, -15.7816982, 15.8036699
12: -11.8803387, 9.8273506, -11.8969126, 9.8282032, -14.9918747, 14.9750443
13: -18.5488148, 6.7333035, -18.5563164, 6.7251682, -16.5730400, 16.6237984
14: 5.0063210, 36.4177322, 4.9736452, 36.4200592, -26.6756516, 26.6957550
15: -8.7190590, 9.2830095, -8.6904716, 9.2849789, -18.0040379, 17.9734802
16: -16.7134209, 2.5312519, -16.7219715, 2.5350039, -14.7804642, 14.8065109
17: 6.2311926, 30.6604366, 6.2125912, 30.6570415, -17.1976814, 17.2178802
18: -14.4051380, 5.1194611, -14.3932724, 5.1245203, -14.4141445, 14.4026699
19: -20.2774010, -4.3255701, -20.2756119, -4.3223195, -14.5305252, 14.5309486
20: -2.4184926, 11.2257118, -2.4222696, 11.2188244, -12.6078186, 12.6188164
21: -11.0666161, 3.2562687, -11.0716753, 3.2504597, -14.3170757, 14.3279438
22: -3.6963327, 13.1053200, -3.6938438, 13.1053343, -14.9481506, 14.9026680
23: -14.5857983, 0.3429675, -14.5793962, 0.3465786, -14.3169441, 14.2959442
24: -19.9449272, -5.1138191, -19.9325600, -5.1135116, -9.2725029, 9.2583961
25: -5.4499130, 10.8572226, -5.4469652, 10.8595181, -13.8007812, 13.7667961
26: -21.0150719, 1.2085817, -21.0074978, 1.2103772, -19.3396416, 19.2537994
27: -16.0052452, 2.1609652, -16.0084362, 2.1714501, -13.2105560, 13.1944313
28: -12.7920780, 4.6343503, -12.7952499, 4.6409826, -17.4330597, 17.4295998
29: -5.5812979, 11.8863335, -5.5895457, 11.8867149, -14.9478493, 14.9016457
30: -10.0400467, 6.2153807, -10.0456734, 6.2070994, -13.5400238, 13.5457001
31: -10.9748917, 6.9481087, -10.9742136, 6.9496317, -14.6433372, 14.6416168
32: -24.9085140, -4.5664988, -24.9223804, -4.5768242, -13.2583923, 13.2955818
33: -69.3158798, -40.1197433, -69.3125839, -40.1090775, -16.6198425, 16.6328392
34: -53.7558861, -30.9352570, -53.7625999, -30.9172077, -14.1115189, 14.1077576
35: -47.8190765, -26.0756378, -47.8189697, -26.0695915, -12.9890862, 12.9857750
36: -42.8130264, -19.2985840, -42.8215485, -19.2848816, -15.0862503, 15.0604095
37: -86.6970825, -55.5515976, -86.6752930, -55.5462189, -18.9315605, 18.9015465
38: -52.9309845, -24.3515816, -52.9460144, -24.3360596, -18.3084793, 18.3416748
39: -76.5651703, -44.6293678, -76.5581970, -44.6260719, -16.0812531, 16.0687752
40: -67.2470016, -43.5563965, -67.2506714, -43.5374603, -14.3301849, 14.3101883
41: -55.4193878, -32.9878883, -55.4286919, -32.9700470, -16.6708603, 16.6480370
42: -29.4565659, -9.8942566, -29.4659386, -9.8873196, -17.2470512, 17.2288628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4343372, upper bound: 12.5631753
time: 7.24 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4542779, upper bound: 12.5655894
time: 7.33 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -12.1142521, 3.6761909, -12.1179781, 3.6758494, -13.8624229, 13.8786087
1: -3.6568115, 7.3928280, -3.6616268, 7.3960094, -8.4646835, 8.4956436
2: -0.7346563, 13.4229317, -0.7391180, 13.4310684, -13.4230652, 13.4318504
3: -1.1113422, 11.2881222, -1.1205537, 11.3009930, -12.0038795, 11.9850540
4: -11.1026478, 5.4750886, -11.1075268, 5.4815502, -14.6400070, 14.6584778
5: 1.8569393, 17.7308655, 1.8513455, 17.7414684, -15.8845291, 15.8795204
6: -39.9026413, -18.2561378, -39.9177322, -18.2402439, -15.0919685, 15.1505280
7: -3.5536547, 12.2423897, -3.5642319, 12.2504759, -13.5405807, 13.5963898
8: -6.6980085, 8.5631971, -6.7016487, 8.5692863, -12.0881271, 12.0744476
9: -4.7721634, 11.7247143, -4.7818379, 11.7120247, -12.9914932, 12.9882622
10: 1.3394637, 25.7411919, 1.3213081, 25.7364578, -20.8834610, 20.9269943
11: -11.4826279, 4.2917314, -11.4930725, 4.2862501, -15.7688780, 15.7848034
12: -11.8812904, 9.8258543, -11.8950577, 9.8270483, -14.9799385, 14.9825859
13: -18.5418644, 6.7285872, -18.5523014, 6.7249622, -16.5688400, 16.6178055
14: 5.0159798, 36.4419174, 4.9819241, 36.4198990, -26.6643143, 26.7045212
15: -8.6823711, 9.2606735, -8.6874924, 9.2743320, -17.9567032, 17.9481659
16: -16.7175045, 2.5433338, -16.7252445, 2.5346122, -14.7775879, 14.8123779
17: 6.2451153, 30.6534195, 6.2226992, 30.6565018, -17.1917648, 17.1917114
18: -14.3852835, 5.1108222, -14.3890390, 5.1207952, -14.3973618, 14.3959064
19: -20.2679577, -4.3269095, -20.2733383, -4.3230267, -14.5301323, 14.5243340
20: -2.4084053, 11.2100410, -2.4168255, 11.2171354, -12.5958481, 12.5966148
21: -11.0612259, 3.2491462, -11.0691357, 3.2498744, -14.3111000, 14.3182821
22: -3.6898556, 13.0991478, -3.6936746, 13.1038322, -14.9353638, 14.8910789
23: -14.5770473, 0.3388171, -14.5783815, 0.3441896, -14.3146744, 14.2906418
24: -19.9281082, -5.1263466, -19.9305458, -5.1200409, -9.2495308, 9.2508469
25: -5.4385490, 10.8489780, -5.4462128, 10.8555317, -13.7881355, 13.7581329
26: -20.9876556, 1.1927938, -21.0035877, 1.2021561, -19.3167648, 19.2441483
27: -16.0079556, 2.1606231, -16.0082436, 2.1712325, -13.2112503, 13.1928864
28: -12.7918139, 4.6353612, -12.7942867, 4.6413336, -17.4331474, 17.4296474
29: -5.5788155, 11.8822308, -5.5886021, 11.8855247, -14.9401855, 14.8933983
30: -10.0341291, 6.2098932, -10.0430355, 6.2062244, -13.5347900, 13.5369759
31: -10.9733582, 6.9468708, -10.9729710, 6.9499865, -14.6441307, 14.6305923
32: -24.8982925, -4.5986009, -24.9123936, -4.5837183, -13.2523537, 13.2588577
33: -69.3355408, -40.1194420, -69.3121185, -40.1085358, -16.6363831, 16.6309471
34: -53.7781715, -30.9301414, -53.7624702, -30.9156685, -14.1313438, 14.1096802
35: -47.8313446, -26.0738564, -47.8184128, -26.0693054, -12.9988937, 12.9880905
36: -42.8319626, -19.2968178, -42.8206520, -19.2838879, -15.0989990, 15.0587234
37: -86.6768494, -55.5763016, -86.6733704, -55.5592155, -18.8929520, 18.8810349
38: -52.9551315, -24.3474159, -52.9454231, -24.3345528, -18.3193893, 18.3412933
39: -76.5692749, -44.6313324, -76.5574036, -44.6263123, -16.0797844, 16.0667229
40: -67.2608490, -43.5503922, -67.2495499, -43.5358315, -14.3225594, 14.3051414
41: -55.4380379, -32.9814987, -55.4281311, -32.9680862, -16.6779060, 16.6472054
42: -29.4594116, -9.9100237, -29.4666824, -9.8958874, -17.2472954, 17.2198181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4422144, upper bound: 12.5631753
time: 7.67 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4621553, upper bound: 12.5655894
time: 7.77 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -12.1217403, 3.6794915, -12.1191730, 3.6762867, -13.8729095, 13.8821030
1: -3.6576724, 7.3949580, -3.6603441, 7.3962131, -8.4657784, 8.5002441
2: -0.7408388, 13.4297047, -0.7411602, 13.4312592, -13.4314384, 13.4445648
3: -1.1289748, 11.3110218, -1.1291057, 11.3018799, -12.0201111, 12.0166168
4: -11.1062365, 5.4811869, -11.1076641, 5.4828649, -14.6414413, 14.6647263
5: 1.8408852, 17.7513161, 1.8431988, 17.7423515, -15.9002914, 15.9081173
6: -39.9237785, -18.2301216, -39.9273148, -18.2396584, -15.1104164, 15.1928864
7: -3.5831599, 12.2729340, -3.5777242, 12.2513971, -13.5654449, 13.6406250
8: -6.7028136, 8.5671873, -6.7011580, 8.5695028, -12.0904007, 12.0935459
9: -4.7814913, 11.7354650, -4.7821641, 11.7161283, -13.0080109, 12.9974365
10: 1.3213930, 25.7514153, 1.3201113, 25.7413101, -20.9064484, 20.9362564
11: -11.5020809, 4.3122535, -11.4990292, 4.2863417, -15.7884226, 15.8112831
12: -11.8860521, 9.8323364, -11.8970985, 9.8278980, -14.9968567, 14.9789162
13: -18.5514431, 6.7439337, -18.5566998, 6.7250752, -16.5749664, 16.6277390
14: 4.9879150, 36.4557648, 4.9685144, 36.4201279, -26.6909027, 26.7313156
15: -8.7162704, 9.2873096, -8.6877441, 9.2854834, -18.0017548, 17.9750538
16: -16.7239285, 2.5441136, -16.7236252, 2.5350778, -14.7877350, 14.8253517
17: 6.2227855, 30.6671028, 6.2116194, 30.6570988, -17.2113533, 17.2158318
18: -14.4014301, 5.1211395, -14.3894520, 5.1248012, -14.4170799, 14.4050369
19: -20.2789993, -4.3244090, -20.2743301, -4.3220797, -14.5361404, 14.5306816
20: -2.4230499, 11.2246437, -2.4224656, 11.2176743, -12.6092300, 12.6185074
21: -11.0686712, 3.2569180, -11.0699196, 3.2504039, -14.3190746, 14.3268375
22: -3.7028778, 13.1095276, -3.6937263, 13.1058426, -14.9602661, 14.9061203
23: -14.5902767, 0.3442628, -14.5797901, 0.3465934, -14.3194809, 14.2980347
24: -19.9435978, -5.1134367, -19.9310341, -5.1135640, -9.2740898, 9.2598381
25: -5.4521904, 10.8586769, -5.4465928, 10.8595476, -13.8081665, 13.7662010
26: -21.0140419, 1.2106886, -21.0043240, 1.2104874, -19.3434029, 19.2546387
27: -16.0151253, 2.1644797, -16.0083008, 2.1725068, -13.2120514, 13.1982536
28: -12.8003931, 4.6377068, -12.7954273, 4.6417713, -17.4421654, 17.4331341
29: -5.5835910, 11.8879538, -5.5896063, 11.8863792, -14.9551620, 14.9025536
30: -10.0452671, 6.2218623, -10.0465775, 6.2068310, -13.5446930, 13.5537415
31: -10.9825897, 6.9501719, -10.9739723, 6.9502182, -14.6499252, 14.6430931
32: -24.9184608, -4.5753050, -24.9224453, -4.5833087, -13.2683411, 13.2953873
33: -69.3475342, -40.1064529, -69.3122559, -40.1046677, -16.6532936, 16.6415253
34: -53.7833138, -30.9242744, -53.7625885, -30.9140816, -14.1380768, 14.1163177
35: -47.8365974, -26.0681114, -47.8188896, -26.0677757, -13.0086517, 12.9926834
36: -42.8335419, -19.2924232, -42.8213577, -19.2834740, -15.1027527, 15.0642662
37: -86.7077408, -55.5460892, -86.6749878, -55.5445786, -18.9388275, 18.9058723
38: -52.9575539, -24.3438702, -52.9459496, -24.3340950, -18.3237534, 18.3464508
39: -76.5816040, -44.6225357, -76.5579987, -44.6237106, -16.0974426, 16.0741425
40: -67.2680359, -43.5493317, -67.2505264, -43.5345688, -14.3286629, 14.3146820
41: -55.4431763, -32.9796829, -55.4287491, -32.9674759, -16.6828995, 16.6510773
42: -29.4627991, -9.9063950, -29.4661694, -9.8954268, -17.2510223, 17.2268944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4928652, upper bound: 12.5230647
time: 7.24 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4950464, upper bound: 12.5655891
time: 20.98 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -12.1179657, 3.6702101, -12.1210289, 3.6761165, -13.8657150, 13.8741913
1: -3.6665001, 7.3916416, -3.6688938, 7.3967600, -8.4672546, 8.4899387
2: -0.7366000, 13.4234734, -0.7410910, 13.4322758, -13.4245224, 13.4365578
3: -1.1133707, 11.2992773, -1.1212891, 11.3092728, -12.0061035, 11.9900589
4: -11.1095171, 5.4751029, -11.1124296, 5.4828801, -14.6494370, 14.6680679
5: 1.8584347, 17.7367783, 1.8506746, 17.7461472, -15.8877125, 15.8861036
6: -39.9073448, -18.2286205, -39.9183846, -18.2228985, -15.1120529, 15.1463928
7: -3.5542011, 12.2428036, -3.5657644, 12.2525187, -13.5449295, 13.5849800
8: -6.7009921, 8.5638371, -6.7053428, 8.5700321, -12.0900574, 12.0856762
9: -4.7781849, 11.7076607, -4.7886386, 11.7125435, -12.9933777, 12.9848099
10: 1.3262124, 25.7322655, 1.3094935, 25.7373505, -20.8882446, 20.9338226
11: -11.4819994, 4.2856317, -11.4958048, 4.2869806, -15.7689800, 15.7814369
12: -11.8872557, 9.8254871, -11.9019909, 9.8280830, -14.9804688, 14.9918442
13: -18.5492668, 6.7231569, -18.5585918, 6.7267509, -16.5626221, 16.6371803
14: 4.9985428, 36.4191170, 4.9662056, 36.4206161, -26.6641235, 26.7081833
15: -8.6940899, 9.2620163, -8.6953592, 9.2761459, -17.9702358, 17.9573746
16: -16.7241249, 2.5281005, -16.7357521, 2.5351007, -14.7873154, 14.7939873
17: 6.2357817, 30.6552963, 6.2135272, 30.6570473, -17.1890984, 17.2013550
18: -14.3900270, 5.1128874, -14.3941116, 5.1227694, -14.3965988, 14.4023552
19: -20.2710724, -4.3234243, -20.2772007, -4.3198524, -14.5340424, 14.5319786
20: -2.4109406, 11.2247095, -2.4183869, 11.2270737, -12.6073303, 12.6057625
21: -11.0635662, 3.2496257, -11.0736980, 3.2511263, -14.3146925, 14.3233242
22: -3.6847913, 13.0994711, -3.6948247, 13.1075459, -14.9306641, 14.8937912
23: -14.5750704, 0.3428969, -14.5804119, 0.3479910, -14.3212509, 14.2970619
24: -19.9317245, -5.1260514, -19.9334278, -5.1193671, -9.2514534, 9.2521744
25: -5.4432354, 10.8501148, -5.4506197, 10.8560371, -13.7857628, 13.7618446
26: -20.9939098, 1.1937785, -21.0096779, 1.2036183, -19.3205032, 19.2491531
27: -16.0080357, 2.1716795, -16.0094986, 2.1791701, -13.2302132, 13.1826782
28: -12.7903423, 4.6426649, -12.7958918, 4.6469169, -17.4372597, 17.4385567
29: -5.5786252, 11.8824730, -5.5896254, 11.8872910, -14.9439278, 14.8953514
30: -10.0352335, 6.2055149, -10.0461855, 6.2070770, -13.5353889, 13.5313721
31: -10.9702282, 6.9482336, -10.9759827, 6.9525490, -14.6443253, 14.6350327
32: -24.9032059, -4.5674868, -24.9133263, -4.5637345, -13.2687950, 13.2598076
33: -69.3111038, -40.1177559, -69.3133698, -40.1037025, -16.6047478, 16.6290550
34: -53.7611427, -30.9210014, -53.7629929, -30.9069481, -14.1283379, 14.1058960
35: -47.8178635, -26.0725880, -47.8189545, -26.0651627, -12.9900665, 12.9897957
36: -42.8183365, -19.2858925, -42.8209915, -19.2747498, -15.0995598, 15.0622330
37: -86.6696930, -55.5746536, -86.6746674, -55.5564728, -18.8884888, 18.8869934
38: -52.9424858, -24.3314400, -52.9459763, -24.3219566, -18.3336639, 18.3370056
39: -76.5549011, -44.6328125, -76.5587769, -44.6253128, -16.0679092, 16.0723572
40: -67.2474213, -43.5435562, -67.2505722, -43.5303879, -14.3430023, 14.2927589
41: -55.4267082, -32.9689980, -55.4290390, -32.9584198, -16.6924362, 16.6434593
42: -29.4638977, -9.8797598, -29.4674110, -9.8770313, -17.2636032, 17.2219658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4561023, upper bound: 12.5233774
time: 17.19 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4591163, upper bound: 12.5667498
time: 10.32 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -12.1254845, 3.6735055, -12.1222000, 3.6765933, -13.8762169, 13.8776894
1: -3.6673741, 7.3937798, -3.6676292, 7.3969722, -8.4683456, 8.4946079
2: -0.7428075, 13.4302378, -0.7431380, 13.4325008, -13.4328690, 13.4492912
3: -1.1310205, 11.3221741, -1.1298431, 11.3101463, -12.0223198, 12.0216370
4: -11.1131096, 5.4811902, -11.1125603, 5.4841919, -14.6508789, 14.6743011
5: 1.8423886, 17.7572079, 1.8425283, 17.7470055, -15.9046173, 15.9146795
6: -39.9284630, -18.2026215, -39.9279594, -18.2223282, -15.1305046, 15.1887474
7: -3.5836933, 12.2733774, -3.5792766, 12.2534475, -13.5698242, 13.6292343
8: -6.7058177, 8.5678358, -6.7048273, 8.5702028, -12.0923462, 12.1047764
9: -4.7874999, 11.7184210, -4.7889624, 11.7166204, -13.0098915, 12.9939690
10: 1.3081274, 25.7424603, 1.3082509, 25.7421703, -20.9112701, 20.9430618
11: -11.5014610, 4.3061619, -11.5017567, 4.2870770, -15.7885380, 15.8079185
12: -11.8920021, 9.8319635, -11.9040384, 9.8289490, -14.9973526, 14.9881363
13: -18.5588264, 6.7384858, -18.5630054, 6.7269688, -16.5687561, 16.6471100
14: 4.9705515, 36.4329376, 4.9527922, 36.4208565, -26.6907654, 26.7349854
15: -8.7279673, 9.2886343, -8.6956110, 9.2872925, -18.0152588, 17.9842453
16: -16.7305775, 2.5288684, -16.7341557, 2.5355515, -14.7974625, 14.8069572
17: 6.2134247, 30.6689777, 6.2024469, 30.6576824, -17.2087021, 17.2254906
18: -14.4061747, 5.1231613, -14.3945637, 5.1267776, -14.4163170, 14.4115009
19: -20.2821178, -4.3209257, -20.2781925, -4.3189125, -14.5400810, 14.5383263
20: -2.4255645, 11.2392931, -2.4240148, 11.2276077, -12.6207161, 12.6276245
21: -11.0709915, 3.2573836, -11.0744915, 3.2516494, -14.3226414, 14.3318748
22: -3.6978254, 13.1098623, -3.6948662, 13.1095753, -14.9555588, 14.9088402
23: -14.5883036, 0.3483124, -14.5818214, 0.3503883, -14.3260193, 14.3044586
24: -19.9472275, -5.1131396, -19.9339237, -5.1128693, -9.2759857, 9.2611427
25: -5.4568925, 10.8598270, -5.4509687, 10.8600578, -13.8058357, 13.7698975
26: -21.0203285, 1.2116427, -21.0104599, 1.2119508, -19.3471489, 19.2596283
27: -16.0152168, 2.1755254, -16.0095749, 2.1804209, -13.2309952, 13.1880493
28: -12.7989378, 4.6450467, -12.7970066, 4.6473870, -17.4463253, 17.4420528
29: -5.5833845, 11.8881979, -5.5906410, 11.8881397, -14.9589310, 14.9044495
30: -10.0463800, 6.2174640, -10.0497551, 6.2076688, -13.5452957, 13.5481415
31: -10.9794197, 6.9515271, -10.9770155, 6.9527578, -14.6501350, 14.6475296
32: -24.9233780, -4.5441904, -24.9233818, -4.5633321, -13.2847595, 13.2963448
33: -69.3231125, -40.1047134, -69.3134918, -40.0998573, -16.6216698, 16.6395950
34: -53.7662468, -30.9151306, -53.7631149, -30.9053955, -14.1350479, 14.1125374
35: -47.8231239, -26.0668011, -47.8193855, -26.0636196, -12.9998207, 12.9943962
36: -42.8199615, -19.2814026, -42.8216782, -19.2743225, -15.1033173, 15.0677872
37: -86.7006378, -55.5444183, -86.6763000, -55.5418625, -18.9344101, 18.9117775
38: -52.9448853, -24.3278713, -52.9465256, -24.3214836, -18.3380585, 18.3421707
39: -76.5673065, -44.6239929, -76.5593338, -44.6226845, -16.0855484, 16.0797768
40: -67.2546082, -43.5424690, -67.2515564, -43.5291748, -14.3491020, 14.3022861
41: -55.4318161, -32.9671936, -55.4296303, -32.9578056, -16.6974487, 16.6473503
42: -29.4673023, -9.8761024, -29.4669323, -9.8765488, -17.2673416, 17.2290573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4900250, upper bound: 12.5233774
time: 7.67 seconds

## Relational analysis of IS_A1_A2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4929624, upper bound: 12.5667498
time: 7.39 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -12.1166553, 3.6756177, -12.1195059, 3.6763444, -13.8656845, 13.8790970
1: -3.6620793, 7.3952579, -3.6656663, 7.3970556, -8.4694138, 8.5015373
2: -0.7355334, 13.4254379, -0.7398018, 13.4325190, -13.4251404, 13.4385605
3: -1.1134592, 11.2969017, -1.1212466, 11.3061619, -12.0120697, 11.9918747
4: -11.1046038, 5.4761286, -11.1087170, 5.4831614, -14.6472778, 14.6680565
5: 1.8580184, 17.7367020, 1.8507013, 17.7450333, -15.8870144, 15.8860006
6: -39.9145966, -18.2350826, -39.9185486, -18.2277298, -15.1163483, 15.1458893
7: -3.5556974, 12.2460489, -3.5651975, 12.2525625, -13.5457840, 13.5945168
8: -6.7005453, 8.5649080, -6.7035494, 8.5702152, -12.0909119, 12.0873566
9: -4.7859964, 11.7279491, -4.7906718, 11.7124825, -12.9977951, 12.9988174
10: 1.3121896, 25.7514935, 1.3052402, 25.7372227, -20.8999710, 20.9535828
11: -11.4893446, 4.2915792, -11.4974604, 4.2863560, -15.7757006, 15.7890396
12: -11.8929605, 9.8304634, -11.9022026, 9.8277903, -14.9854393, 14.9957428
13: -18.5519810, 6.7337379, -18.5589771, 6.7267036, -16.5645981, 16.6411362
14: 4.9802065, 36.4570999, 4.9610710, 36.4207153, -26.6793671, 26.7437668
15: -8.6912889, 9.2662964, -8.6926231, 9.2766695, -17.9679585, 17.9589195
16: -16.7346478, 2.5409660, -16.7373981, 2.5351732, -14.7945786, 14.8128548
17: 6.2273870, 30.6619511, 6.2126136, 30.6570969, -17.2027702, 17.1993256
18: -14.3863049, 5.1145563, -14.3903217, 5.1230583, -14.3995419, 14.4047241
19: -20.2726421, -4.3222575, -20.2759171, -4.3196130, -14.5396843, 14.5316963
20: -2.4155045, 11.2236376, -2.4185879, 11.2259388, -12.6087418, 12.6054573
21: -11.0656338, 3.2502644, -11.0719194, 3.2510550, -14.3166885, 14.3221836
22: -3.6913323, 13.1037169, -3.6946790, 13.1080675, -14.9428024, 14.8972282
23: -14.5795765, 0.3441617, -14.5808105, 0.3479862, -14.3237534, 14.2991638
24: -19.9304047, -5.1256971, -19.9318771, -5.1193905, -9.2530136, 9.2535934
25: -5.4455276, 10.8516045, -5.4502363, 10.8560991, -13.7931366, 13.7612610
26: -20.9929142, 1.1958840, -21.0065613, 1.2037504, -19.3242645, 19.2500000
27: -16.0179291, 2.1751924, -16.0093918, 2.1802058, -13.2316971, 13.1865158
28: -12.7986774, 4.6460176, -12.7960491, 4.6477418, -17.4464188, 17.4420662
29: -5.5809193, 11.8840952, -5.5897236, 11.8869858, -14.9512482, 14.8962135
30: -10.0403976, 6.2119694, -10.0471287, 6.2068033, -13.5400658, 13.5394058
31: -10.9778919, 6.9503050, -10.9757738, 6.9531121, -14.6509247, 14.6365280
32: -24.9131641, -4.5762882, -24.9134331, -4.5702543, -13.2787056, 13.2596092
33: -69.3427734, -40.1044388, -69.3131104, -40.0993385, -16.6381683, 16.6377296
34: -53.7885056, -30.9100266, -53.7629700, -30.9038067, -14.1549034, 14.1144562
35: -47.8353882, -26.0650520, -47.8188477, -26.0633469, -13.0096283, 12.9967461
36: -42.8388443, -19.2796688, -42.8208771, -19.2733383, -15.1160355, 15.0660782
37: -86.6803513, -55.5691910, -86.6743774, -55.5548782, -18.8957710, 18.8913116
38: -52.9690514, -24.3237343, -52.9459686, -24.3199406, -18.3489380, 18.3417816
39: -76.5714035, -44.6259995, -76.5586090, -44.6229858, -16.0840988, 16.0777359
40: -67.2684631, -43.5364838, -67.2504272, -43.5275116, -14.3414650, 14.2972336
41: -55.4504890, -32.9607849, -55.4290771, -32.9558411, -16.7044640, 16.6465034
42: -29.4701118, -9.8918610, -29.4676361, -9.8851328, -17.2676392, 17.2199821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4975057, upper bound: 12.5233774
time: 7.92 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5004658, upper bound: 12.5667498
time: 12.77 seconds

## BFS IS instance: IS_A1_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -12.1241932, 3.6789312, -12.1206703, 3.6768270, -13.8761864, 13.8825760
1: -3.6629465, 7.3973970, -3.6643715, 7.3972673, -8.4705238, 8.5061417
2: -0.7417241, 13.4322014, -0.7418541, 13.4327154, -13.4334984, 13.4512711
3: -1.1310960, 11.3198013, -1.1298237, 11.3070250, -12.0283051, 12.0234413
4: -11.1082077, 5.4822111, -11.1088505, 5.4844751, -14.6487122, 14.6742973
5: 1.8419628, 17.7571602, 1.8425598, 17.7459145, -15.9039516, 15.9146004
6: -39.9357147, -18.2090645, -39.9280930, -18.2271748, -15.1347733, 15.1882477
7: -3.5851748, 12.2766171, -3.5787086, 12.2534981, -13.5706711, 13.6387558
8: -6.7053518, 8.5689220, -6.7030535, 8.5703716, -12.0931854, 12.1064682
9: -4.7953005, 11.7386818, -4.7910056, 11.7165747, -13.0143051, 13.0080147
10: 1.2941113, 25.7616920, 1.3040490, 25.7420444, -20.9229660, 20.9627686
11: -11.5087948, 4.3121047, -11.5034304, 4.2864542, -15.7952490, 15.8155346
12: -11.8976879, 9.8369656, -11.9042349, 9.8286457, -15.0023537, 14.9920273
13: -18.5614834, 6.7491589, -18.5634022, 6.7268910, -16.5706940, 16.6510620
14: 4.9521694, 36.4709396, 4.9476891, 36.4208946, -26.7059326, 26.7704773
15: -8.7252073, 9.2929306, -8.6928844, 9.2878027, -18.0130100, 17.9858150
16: -16.7411003, 2.5417089, -16.7358189, 2.5356390, -14.8047295, 14.8258247
17: 6.2050152, 30.6755943, 6.2015815, 30.6576996, -17.2223434, 17.2234230
18: -14.4024878, 5.1248260, -14.3907499, 5.1270685, -14.4192467, 14.4138641
19: -20.2836952, -4.3197503, -20.2768936, -4.3186798, -14.5456924, 14.5380440
20: -2.4301262, 11.2382307, -2.4242153, 11.2264729, -12.6221199, 12.6273384
21: -11.0730438, 3.2580314, -11.0727224, 3.2515879, -14.3246317, 14.3307533
22: -3.7043464, 13.1140966, -3.6947296, 13.1100769, -14.9677124, 14.9122543
23: -14.5927858, 0.3495951, -14.5822277, 0.3504128, -14.3285217, 14.3065262
24: -19.9458923, -5.1127787, -19.9323654, -5.1129093, -9.2775612, 9.2625732
25: -5.4591331, 10.8613024, -5.4506092, 10.8600903, -13.8131943, 13.7693253
26: -21.0193176, 1.2137601, -21.0072899, 1.2120831, -19.3508644, 19.2604980
27: -16.0251255, 2.1790340, -16.0094299, 2.1814423, -13.2324982, 13.1918869
28: -12.8072462, 4.6484051, -12.7971954, 4.6481857, -17.4554329, 17.4456005
29: -5.5856924, 11.8898096, -5.5907021, 11.8878241, -14.9662552, 14.9053345
30: -10.0515604, 6.2239285, -10.0506601, 6.2073946, -13.5499725, 13.5561714
31: -10.9871273, 6.9536123, -10.9768209, 6.9533367, -14.6567116, 14.6490135
32: -24.9333496, -4.5529985, -24.9235020, -4.5698366, -13.2946968, 13.2961349
33: -69.3547592, -40.0914688, -69.3131866, -40.0954437, -16.6550446, 16.6482811
34: -53.7936554, -30.9041958, -53.7630997, -30.9022770, -14.1616096, 14.1210976
35: -47.8406563, -26.0592613, -47.8192940, -26.0618324, -13.0193825, 13.0013657
36: -42.8404007, -19.2752380, -42.8215637, -19.2728920, -15.1198006, 15.0716133
37: -86.7112885, -55.5389252, -86.6760635, -55.5402222, -18.9416580, 18.9161263
38: -52.9714203, -24.3200970, -52.9464760, -24.3195915, -18.3533325, 18.3469696
39: -76.5837708, -44.6171875, -76.5591278, -44.6203270, -16.1017303, 16.0851364
40: -67.2756729, -43.5354309, -67.2513657, -43.5262909, -14.3475571, 14.3067627
41: -55.4556427, -32.9589157, -55.4296722, -32.9552689, -16.7094994, 16.6503944
42: -29.4735126, -9.8882160, -29.4671288, -9.8846655, -17.2713547, 17.2270737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=89, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A1_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5309294, upper bound: 12.5233774
time: 21.91 seconds

## Relational analysis of IS_A1_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5338450, upper bound: 12.5667498
time: 13.75 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.1105757, 3.6488662, -12.1005373, 3.6457498, -13.8461304, 13.8342628
1: -3.6546011, 7.3809090, -3.6456404, 7.3744822, -8.4615822, 8.4651585
2: -0.7479553, 13.4168158, -0.7186192, 13.4066191, -13.4222412, 13.3996391
3: -1.1165295, 11.2857733, -1.1052860, 11.2798214, -11.9898605, 11.9733734
4: -11.0874119, 5.4485521, -11.0802536, 5.4563808, -14.6187363, 14.6072273
5: 1.8528414, 17.7341919, 1.8685827, 17.7278671, -15.8750257, 15.8656092
6: -39.8515015, -18.2870617, -39.8587990, -18.2488232, -15.0754280, 15.0642548
7: -3.5613554, 12.2299366, -3.5324516, 12.2257805, -13.5768433, 13.5343933
8: -6.7132916, 8.5550566, -6.6930699, 8.5515270, -12.0813446, 12.0756645
9: -4.7476311, 11.6951790, -4.7655964, 11.6757736, -12.9330215, 12.9590492
10: 1.3435388, 25.7449188, 1.3487973, 25.7049484, -20.8443451, 20.8579483
11: -11.4784927, 4.2903199, -11.4744892, 4.2825141, -15.7610073, 15.7648087
12: -11.8672180, 9.8428173, -11.8533592, 9.8103304, -14.9534073, 14.9849968
13: -18.5322208, 6.7130747, -18.5286274, 6.7101884, -16.6051483, 16.5019722
14: 5.0172710, 36.4329300, 5.0393572, 36.3891907, -26.6439209, 26.6443253
15: -8.6518736, 9.2103310, -8.6853819, 9.2416115, -17.8934860, 17.8957138
16: -16.7198219, 2.5359240, -16.7054749, 2.5076435, -14.7342834, 14.7900658
17: 6.2624907, 30.6561966, 6.2870860, 30.6245842, -17.1532593, 17.1269836
18: -14.3555737, 5.1043463, -14.3625631, 5.1050205, -14.3465652, 14.3593102
19: -20.2531071, -4.3376756, -20.2507744, -4.3356633, -14.4973145, 14.4972343
20: -2.3942621, 11.2042294, -2.3808358, 11.2018814, -12.5735970, 12.5697746
21: -11.0496283, 3.2574406, -11.0427198, 3.2406049, -14.2902336, 14.3001604
22: -3.6718259, 13.0955143, -3.6608381, 13.0635891, -14.8596764, 14.8905220
23: -14.5478973, 0.3009212, -14.5618095, 0.3147159, -14.2464752, 14.2654076
24: -19.9278297, -5.1290812, -19.9259186, -5.1247082, -9.2390251, 9.2392998
25: -5.4374628, 10.8520555, -5.4276934, 10.8360882, -13.7450790, 13.7556648
26: -20.9587097, 1.1911261, -20.9539223, 1.1393211, -19.2266273, 19.2681503
27: -16.0145798, 2.1614695, -15.9926224, 2.1571748, -13.1530151, 13.1872025
28: -12.7672453, 4.6011505, -12.7706585, 4.6077476, -17.3749924, 17.3718090
29: -5.5366478, 11.8722868, -5.5457506, 11.8351269, -14.8467178, 14.8878555
30: -10.0291271, 6.2199678, -10.0191650, 6.1904516, -13.5089340, 13.5285187
31: -10.9553585, 6.9484658, -10.9512634, 6.9484682, -14.5989647, 14.6150818
32: -24.8909264, -4.6001425, -24.8811398, -4.5903816, -13.2441559, 13.2335968
33: -69.3171082, -40.1468964, -69.2766953, -40.1447678, -16.6101227, 16.5818901
34: -53.7585526, -30.9384499, -53.7302933, -30.9400711, -14.1023598, 14.0968552
35: -47.8161583, -26.0835266, -47.7992325, -26.0762711, -12.9870949, 12.9673882
36: -42.8155212, -19.2934647, -42.7781219, -19.3003006, -15.0657845, 15.0448341
37: -86.6679001, -55.5790710, -86.6525879, -55.5912399, -18.8538361, 18.8715744
38: -52.9216232, -24.3654785, -52.8914642, -24.3450298, -18.3045654, 18.2660294
39: -76.5350494, -44.6563568, -76.5290070, -44.6496429, -16.0296364, 16.0321922
40: -67.2637863, -43.5460663, -67.2168579, -43.5598984, -14.2437515, 14.3058376
41: -55.4403992, -32.9684410, -55.4111443, -32.9804764, -16.6294289, 16.6639633
42: -29.4540367, -9.9154320, -29.4511604, -9.9122028, -17.1835403, 17.2333870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4370200
time: 11.97 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4619806
time: 8.80 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1117315, 3.6493416, -12.1080341, 3.6490636, -13.8496208, 13.8447685
1: -3.6533022, 7.3811111, -3.6465111, 7.3766336, -8.4662476, 8.4662743
2: -0.7499939, 13.4170465, -0.7248098, 13.4133863, -13.4349518, 13.4079704
3: -1.1250951, 11.2866507, -1.1229337, 11.3027143, -12.0214462, 11.9896145
4: -11.0875425, 5.4498792, -11.0838814, 5.4624825, -14.6249542, 14.6086693
5: 1.8446856, 17.7350807, 1.8525319, 17.7483253, -15.9036398, 15.8825493
6: -39.8610840, -18.2864704, -39.8798943, -18.2228050, -15.1177979, 15.0827026
7: -3.5748656, 12.2308435, -3.5619268, 12.2563524, -13.6210632, 13.5593185
8: -6.7127752, 8.5552464, -6.6978588, 8.5555172, -12.1004181, 12.0779457
9: -4.7479439, 11.6992531, -4.7749124, 11.6865101, -12.9421883, 12.9755592
10: 1.3423352, 25.7497654, 1.3307295, 25.7151413, -20.8535690, 20.8809433
11: -11.4844379, 4.2904139, -11.4939003, 4.3030519, -15.7874899, 15.7843142
12: -11.8692474, 9.8436651, -11.8581028, 9.8168030, -14.9497185, 15.0019112
13: -18.5366287, 6.7132220, -18.5381508, 6.7255797, -16.6150856, 16.5080986
14: 5.0038567, 36.4331589, 5.0113153, 36.4030037, -26.6706848, 26.6708832
15: -8.6520920, 9.2215014, -8.7192707, 9.2682390, -17.9203300, 17.9407730
16: -16.7182083, 2.5363791, -16.7118530, 2.5084496, -14.7472458, 14.8002014
17: 6.2514248, 30.6567841, 6.2647829, 30.6382713, -17.1773605, 17.1465836
18: -14.3560228, 5.1083603, -14.3787184, 5.1153278, -14.3557205, 14.3790512
19: -20.2540817, -4.3367434, -20.2618294, -4.3331566, -14.5036430, 14.5032463
20: -2.3999050, 11.2047710, -2.3954906, 11.2164888, -12.5954781, 12.5831413
21: -11.0504150, 3.2579613, -11.0501356, 3.2483540, -14.2987690, 14.3080969
22: -3.6718731, 13.0975161, -3.6738136, 13.0739717, -14.8747253, 14.9154625
23: -14.5492964, 0.3033586, -14.5750446, 0.3201337, -14.2538528, 14.2701836
24: -19.9283447, -5.1225848, -19.9414234, -5.1117992, -9.2480011, 9.2638321
25: -5.4378433, 10.8560658, -5.4413056, 10.8457985, -13.7531357, 13.7757416
26: -20.9594688, 1.1994896, -20.9803581, 1.1572213, -19.2370834, 19.2948227
27: -16.0146236, 2.1626964, -15.9998045, 2.1610165, -13.1583633, 13.1879921
28: -12.7684040, 4.6016297, -12.7792015, 4.6101117, -17.3785152, 17.3808308
29: -5.5376329, 11.8731318, -5.5505209, 11.8408442, -14.8558807, 14.9028397
30: -10.0326834, 6.2205749, -10.0302973, 6.2023964, -13.5257301, 13.5384407
31: -10.9563789, 6.9486814, -10.9604597, 6.9517646, -14.6114731, 14.6208725
32: -24.9009895, -4.5997314, -24.9013233, -4.5670896, -13.2806740, 13.2495575
33: -69.3171844, -40.1430473, -69.2886658, -40.1318474, -16.6206741, 16.5987816
34: -53.7586632, -30.9368916, -53.7353745, -30.9342384, -14.1090164, 14.1035767
35: -47.8166046, -26.0820560, -47.8044815, -26.0704994, -12.9916801, 12.9771538
36: -42.8162270, -19.2930489, -42.7797089, -19.2958107, -15.0713425, 15.0485954
37: -86.6695175, -55.5644226, -86.6835098, -55.5610390, -18.8786621, 18.9174652
38: -52.9221840, -24.3650532, -52.8938179, -24.3414650, -18.3097534, 18.2704086
39: -76.5355911, -44.6537704, -76.5414276, -44.6408348, -16.0370636, 16.0498352
40: -67.2647400, -43.5448265, -67.2240372, -43.5588303, -14.2532921, 14.3119106
41: -55.4409943, -32.9678345, -55.4163017, -32.9786034, -16.6332703, 16.6689682
42: -29.4535141, -9.9149685, -29.4545269, -9.9085693, -17.1906013, 17.2371330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=96, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4705595
time: 6.74 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4965453
time: 8.06 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -12.1141081, 3.6446033, -12.1037531, 3.6443036, -13.8469086, 13.8324051
1: -3.6639585, 7.3790674, -3.6532459, 7.3759208, -8.4666595, 8.4569855
2: -0.7503281, 13.4171114, -0.7201712, 13.4081411, -13.4282990, 13.3996887
3: -1.1174127, 11.2955418, -1.1071659, 11.2895012, -11.9878387, 11.9826412
4: -11.0943518, 5.4500618, -11.0851011, 5.4562397, -14.6307068, 14.6142578
5: 1.8524332, 17.7388992, 1.8698378, 17.7337322, -15.8812990, 15.8690615
6: -39.8456841, -18.2652531, -39.8699112, -18.2258053, -15.0645180, 15.0910988
7: -3.5614386, 12.2292347, -3.5344393, 12.2289581, -13.5749664, 13.5291901
8: -6.7168980, 8.5554895, -6.6961031, 8.5524797, -12.0939789, 12.0761871
9: -4.7531118, 11.6757412, -4.7729445, 11.6787119, -12.9429092, 12.9476051
10: 1.3346043, 25.7267513, 1.3326120, 25.7150879, -20.8655624, 20.8483353
11: -11.4789677, 4.2850404, -11.4760838, 4.2824535, -15.7614212, 15.7611237
12: -11.8706093, 9.8391056, -11.8628664, 9.8147173, -14.9630623, 14.9850922
13: -18.5381718, 6.7051001, -18.5364151, 6.7145672, -16.6272583, 16.4929848
14: 5.0049114, 36.3961143, 5.0185642, 36.4039574, -26.6774445, 26.6143112
15: -8.6611214, 9.2095146, -8.6956463, 9.2453890, -17.9065094, 17.9051609
16: -16.7266235, 2.5238988, -16.7157516, 2.5048931, -14.7330704, 14.7823792
17: 6.2585559, 30.6504631, 6.2724981, 30.6327705, -17.1475639, 17.1396523
18: -14.3619766, 5.1058321, -14.3660660, 5.1075630, -14.3541374, 14.3574295
19: -20.2555199, -4.3350301, -20.2553997, -4.3316364, -14.4994278, 14.5067062
20: -2.3923929, 11.2148952, -2.3867855, 11.2158146, -12.5812454, 12.5827141
21: -11.0527601, 3.2581074, -11.0465384, 3.2416556, -14.2944155, 14.3046455
22: -3.6670427, 13.0971127, -3.6616957, 13.0660295, -14.8523293, 14.8958702
23: -14.5469427, 0.3046780, -14.5628452, 0.3188200, -14.2522850, 14.2725563
24: -19.9311943, -5.1285715, -19.9290867, -5.1242237, -9.2407990, 9.2407761
25: -5.4404860, 10.8515701, -5.4337473, 10.8382568, -13.7406387, 13.7615204
26: -20.9649391, 1.1909676, -20.9600544, 1.1418340, -19.2299957, 19.2734985
27: -16.0064468, 2.1686285, -16.0021095, 2.1690125, -13.1447868, 13.2041321
28: -12.7614021, 4.6059170, -12.7766056, 4.6159105, -17.3773117, 17.3825226
29: -5.5359402, 11.8724871, -5.5472431, 11.8369179, -14.8417816, 14.8985214
30: -10.0299320, 6.2144346, -10.0226192, 6.1923857, -13.5081253, 13.5243568
31: -10.9523144, 6.9507132, -10.9541187, 6.9501395, -14.5991745, 14.6194878
32: -24.8827515, -4.5750380, -24.8951855, -4.5643468, -13.2325630, 13.2625847
33: -69.2871552, -40.1439896, -69.2834473, -40.1411400, -16.5775757, 16.5808945
34: -53.7320747, -30.9297829, -53.7402039, -30.9309063, -14.0731087, 14.1193008
35: -47.7993164, -26.0810680, -47.8031273, -26.0733109, -12.9722862, 12.9750404
36: -42.7955322, -19.2845230, -42.7848740, -19.2890739, -15.0530701, 15.0616722
37: -86.6590424, -55.5774765, -86.6555862, -55.5884666, -18.8554573, 18.8714676
38: -52.8961220, -24.3528061, -52.9048080, -24.3291359, -18.2854080, 18.2951736
39: -76.5208740, -44.6561012, -76.5302658, -44.6503677, -16.0223427, 16.0332413
40: -67.2441330, -43.5407028, -67.2241211, -43.5529785, -14.2363281, 14.3213043
41: -55.4181137, -32.9587860, -55.4228745, -32.9679489, -16.6132393, 16.6908989
42: -29.4495163, -9.8900795, -29.4608803, -9.8883648, -17.1775131, 17.2578392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5621836, upper bound: 12.4138692
time: 8.22 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664146, upper bound: 12.4572936
time: 7.80 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -12.1152802, 3.6450996, -12.1112604, 3.6475959, -13.8503799, 13.8428917
1: -3.6626728, 7.3792839, -3.6541073, 7.3780384, -8.4713097, 8.4580917
2: -0.7523797, 13.4173212, -0.7263696, 13.4148874, -13.4410324, 13.4080353
3: -1.1259843, 11.2964220, -1.1247996, 11.3123837, -12.0194168, 11.9988613
4: -11.0944767, 5.4513712, -11.0887194, 5.4623218, -14.6369705, 14.6156998
5: 1.8442602, 17.7397652, 1.8537874, 17.7541924, -15.9099321, 15.8859777
6: -39.8552589, -18.2646313, -39.8910408, -18.1997871, -15.1068916, 15.1095505
7: -3.5749445, 12.2301731, -3.5639398, 12.2594872, -13.6192245, 13.5541039
8: -6.7163992, 8.5556631, -6.7009220, 8.5564833, -12.1130791, 12.0784531
9: -4.7534332, 11.6798248, -4.7822676, 11.6894360, -12.9520683, 12.9641228
10: 1.3334265, 25.7315865, 1.3145599, 25.7252693, -20.8747787, 20.8713608
11: -11.4849291, 4.2851191, -11.4955349, 4.3029456, -15.7878742, 15.7806540
12: -11.8726683, 9.8399668, -11.8676348, 9.8212013, -14.9593620, 15.0020142
13: -18.5425453, 6.7052593, -18.5459671, 6.7299447, -16.6371689, 16.4990768
14: 4.9915037, 36.3962975, 4.9904928, 36.4177246, -26.7042236, 26.6409073
15: -8.6613607, 9.2206879, -8.7295399, 9.2720118, -17.9333725, 17.9502277
16: -16.7250137, 2.5243459, -16.7222023, 2.5056982, -14.7460518, 14.7925301
17: 6.2475119, 30.6510544, 6.2501593, 30.6464081, -17.1716995, 17.1592560
18: -14.3624105, 5.1098642, -14.3821936, 5.1178865, -14.3632908, 14.3771706
19: -20.2564812, -4.3341122, -20.2664490, -4.3291302, -14.5057678, 14.5127029
20: -2.3980470, 11.2154140, -2.4014025, 11.2304115, -12.6031189, 12.5960960
21: -11.0535412, 3.2586410, -11.0539646, 3.2494411, -14.3029823, 14.3126059
22: -3.6670830, 13.0991135, -3.6746917, 13.0764046, -14.8673668, 14.9207573
23: -14.5483236, 0.3070846, -14.5760803, 0.3242488, -14.2596970, 14.2773590
24: -19.9316864, -5.1220665, -19.9445724, -5.1113319, -9.2497864, 9.2653008
25: -5.4408445, 10.8555851, -5.4473677, 10.8479519, -13.7486649, 13.7815742
26: -20.9656906, 1.1993372, -20.9864635, 1.1597722, -19.2404785, 19.3001709
27: -16.0064926, 2.1698759, -16.0093002, 2.1728618, -13.1501732, 13.2049217
28: -12.7625446, 4.6063490, -12.7851963, 4.6182632, -17.3808079, 17.3915443
29: -5.5369482, 11.8733330, -5.5520134, 11.8426323, -14.8508720, 14.9135246
30: -10.0334740, 6.2150602, -10.0337620, 6.2043419, -13.5249176, 13.5342751
31: -10.9533472, 6.9509368, -10.9633198, 6.9534512, -14.6116638, 14.6252785
32: -24.8928108, -4.5746231, -24.9153881, -4.5410752, -13.2690964, 13.2785378
33: -69.2872238, -40.1401329, -69.2954102, -40.1281433, -16.5881348, 16.5978241
34: -53.7322083, -30.9282303, -53.7453156, -30.9250546, -14.0797691, 14.1260300
35: -47.7997055, -26.0795288, -47.8083839, -26.0675392, -12.9769020, 12.9848289
36: -42.7962151, -19.2841225, -42.7864304, -19.2846069, -15.0585976, 15.0654449
37: -86.6606750, -55.5628395, -86.6865311, -55.5582275, -18.8803062, 18.9173737
38: -52.8967133, -24.3523808, -52.9071732, -24.3255520, -18.2905998, 18.2995758
39: -76.5214539, -44.6535110, -76.5426025, -44.6415634, -16.0297508, 16.0508804
40: -67.2451019, -43.5394974, -67.2312469, -43.5518875, -14.2458420, 14.3273830
41: -55.4187393, -32.9581909, -55.4280319, -32.9660797, -16.6171074, 16.6959534
42: -29.4489975, -9.8896093, -29.4642582, -9.8847370, -17.1845970, 17.2616081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=96, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5220173, upper bound: 12.4882732
time: 6.84 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4925248
time: 7.06 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.1125860, 3.6448579, -12.1024170, 3.6497269, -13.8518066, 13.8323555
1: -3.6607151, 7.3793573, -3.6488357, 7.3795180, -8.4782410, 8.4591694
2: -0.7490561, 13.4173231, -0.7190968, 13.4100895, -13.4302826, 13.4003029
3: -1.1173868, 11.2924061, -1.1072416, 11.2871323, -11.9896736, 11.9886131
4: -11.0906353, 5.4503570, -11.0801954, 5.4572225, -14.6307068, 14.6121140
5: 1.8524623, 17.7378025, 1.8694057, 17.7336578, -15.8811951, 15.8683968
6: -39.8458366, -18.2700348, -39.8771706, -18.2322674, -15.0640297, 15.0953789
7: -3.5608718, 12.2293005, -3.5359182, 12.2322035, -13.5845108, 13.5300713
8: -6.7151408, 8.5556583, -6.6956329, 8.5535679, -12.0956764, 12.0770206
9: -4.7551270, 11.6756897, -4.7807722, 11.6989708, -12.9569244, 12.9520073
10: 1.3303871, 25.7266350, 1.3186083, 25.7342777, -20.8852768, 20.8600464
11: -11.4806595, 4.2844038, -11.4834261, 4.2883902, -15.7690496, 15.7678299
12: -11.8708134, 9.8388062, -11.8685541, 9.8197289, -14.9669228, 14.9900703
13: -18.5384960, 6.7049747, -18.5390358, 6.7252488, -16.6312180, 16.4949074
14: 4.9997864, 36.3961639, 5.0001621, 36.4419136, -26.7130585, 26.6294937
15: -8.6584129, 9.2100058, -8.6928921, 9.2497158, -17.9081287, 17.9028969
16: -16.7282906, 2.5239925, -16.7262535, 2.5177646, -14.7518845, 14.7896271
17: 6.2576523, 30.6505032, 6.2640972, 30.6393814, -17.1455116, 17.1533203
18: -14.3581657, 5.1061444, -14.3623209, 5.1092467, -14.3565216, 14.3603764
19: -20.2542591, -4.3348050, -20.2569695, -4.3304777, -14.4991608, 14.5124016
20: -2.3926084, 11.2137423, -2.3913517, 11.2147331, -12.5809631, 12.5841370
21: -11.0509987, 3.2580500, -11.0485687, 3.2422905, -14.2932892, 14.3066187
22: -3.6669109, 13.0976315, -3.6682515, 13.0702362, -14.8557587, 14.9080124
23: -14.5473337, 0.3046837, -14.5673532, 0.3201017, -14.2543755, 14.2750664
24: -19.9296360, -5.1286149, -19.9277878, -5.1238527, -9.2422142, 9.2423286
25: -5.4401035, 10.8516169, -5.4360085, 10.8397255, -13.7400360, 13.7689056
26: -20.9618301, 1.1911178, -20.9590626, 1.1439638, -19.2308311, 19.2772827
27: -16.0063152, 2.1696687, -16.0119877, 2.1725020, -13.1486320, 13.2056236
28: -12.7615538, 4.6067128, -12.7849426, 4.6192799, -17.3808327, 17.3916550
29: -5.5360012, 11.8721676, -5.5495567, 11.8385162, -14.8426361, 14.9058304
30: -10.0308475, 6.2141571, -10.0278034, 6.1988778, -13.5161591, 13.5290146
31: -10.9520845, 6.9513011, -10.9618149, 6.9522181, -14.6006737, 14.6260872
32: -24.8828621, -4.5815210, -24.9051628, -4.5731530, -13.2323570, 13.2724915
33: -69.2867889, -40.1395836, -69.3151093, -40.1278458, -16.5862656, 16.6143227
34: -53.7321014, -30.9266834, -53.7676392, -30.9198990, -14.0816841, 14.1458511
35: -47.7992096, -26.0792904, -47.8206711, -26.0657692, -12.9792328, 12.9946060
36: -42.7953796, -19.2831383, -42.8052979, -19.2829361, -15.0568733, 15.0781555
37: -86.6587830, -55.5758514, -86.6662598, -55.5829582, -18.8597679, 18.8787231
38: -52.8960609, -24.3508186, -52.9313736, -24.3213978, -18.2902145, 18.3104324
39: -76.5206757, -44.6537437, -76.5467300, -44.6435623, -16.0276985, 16.0494232
40: -67.2439728, -43.5378265, -67.2451477, -43.5459213, -14.2407913, 14.3197613
41: -55.4181900, -32.9562263, -55.4466858, -32.9597015, -16.6162720, 16.7029839
42: -29.4497223, -9.8981886, -29.4670868, -9.9004869, -17.1755295, 17.2618866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5220173, upper bound: 12.4948017
time: 6.21 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4990493
time: 23.20 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -12.1137428, 3.6453204, -12.1099567, 3.6530390, -13.8553009, 13.8428345
1: -3.6594377, 7.3795805, -3.6496959, 7.3816481, -8.4828453, 8.4602909
2: -0.7511106, 13.4175472, -0.7253023, 13.4168644, -13.4430008, 13.4086609
3: -1.1259462, 11.2933016, -1.1248916, 11.3100224, -12.0212326, 12.0048599
4: -11.0907574, 5.4516582, -11.0838146, 5.4633188, -14.6369476, 14.6135559
5: 1.8443007, 17.7386665, 1.8533444, 17.7541122, -15.9098110, 15.8853226
6: -39.8554382, -18.2694511, -39.8983002, -18.2062378, -15.1063843, 15.1138420
7: -3.5743620, 12.2302151, -3.5654366, 12.2627659, -13.6287308, 13.5549660
8: -6.7146254, 8.5558472, -6.7004461, 8.5575447, -12.1147652, 12.0792694
9: -4.7554469, 11.6797581, -4.7900887, 11.7097178, -12.9660835, 12.9685211
10: 1.3291926, 25.7314682, 1.3005104, 25.7444420, -20.8944855, 20.8830566
11: -11.4866238, 4.2844906, -11.5028591, 4.3089037, -15.7955275, 15.7873497
12: -11.8728580, 9.8396740, -11.8732948, 9.8261719, -14.9632492, 15.0069885
13: -18.5429115, 6.7051334, -18.5486202, 6.7405787, -16.6411285, 16.5010262
14: 4.9864006, 36.3963776, 4.9721451, 36.4557800, -26.7397842, 26.6561050
15: -8.6586218, 9.2211637, -8.7267809, 9.2763147, -17.9349365, 17.9479446
16: -16.7266636, 2.5244086, -16.7327309, 2.5185263, -14.7648697, 14.7997704
17: 6.2465987, 30.6510811, 6.2417526, 30.6530647, -17.1696243, 17.1729088
18: -14.3585968, 5.1101599, -14.3784981, 5.1195364, -14.3656693, 14.3800964
19: -20.2552185, -4.3338647, -20.2680168, -4.3279762, -14.5055084, 14.5183334
20: -2.3982463, 11.2142725, -2.4059541, 11.2293406, -12.6028519, 12.5975075
21: -11.0517702, 3.2585638, -11.0560045, 3.2500551, -14.3018255, 14.3145685
22: -3.6669719, 13.0996342, -3.6812470, 13.0806484, -14.8708191, 14.9329071
23: -14.5487375, 0.3071036, -14.5805702, 0.3255365, -14.2617569, 14.2798424
24: -19.9301376, -5.1221199, -19.9432526, -5.1109481, -9.2511902, 9.2668686
25: -5.4404697, 10.8556032, -5.4496355, 10.8494072, -13.7481003, 13.7889595
26: -20.9625187, 1.1994476, -20.9854660, 1.1618433, -19.2412872, 19.3039017
27: -16.0064030, 2.1709156, -16.0191727, 2.1763616, -13.1540146, 13.2064285
28: -12.7627020, 4.6071720, -12.7934952, 4.6216412, -17.3843422, 17.4006672
29: -5.5370154, 11.8730268, -5.5543065, 11.8442535, -14.8517609, 14.9208298
30: -10.0343666, 6.2147522, -10.0389242, 6.2108264, -13.5329514, 13.5389175
31: -10.9531374, 6.9515247, -10.9710188, 6.9554977, -14.6131668, 14.6318855
32: -24.8929253, -4.5811214, -24.9253159, -4.5498962, -13.2689018, 13.2885056
33: -69.2869186, -40.1357422, -69.3270798, -40.1148758, -16.5968170, 16.6312370
34: -53.7321968, -30.9251251, -53.7727280, -30.9140854, -14.0883141, 14.1525536
35: -47.7996292, -26.0777550, -47.8259621, -26.0599861, -12.9838486, 13.0043716
36: -42.7961197, -19.2827034, -42.8069077, -19.2784271, -15.0624466, 15.0819168
37: -86.6604004, -55.5612183, -86.6971893, -55.5527496, -18.8846169, 18.9246140
38: -52.8966675, -24.3503780, -52.9337540, -24.3178577, -18.2953796, 18.3148041
39: -76.5212402, -44.6511536, -76.5591278, -44.6347427, -16.0351257, 16.0670738
40: -67.2449188, -43.5366058, -67.2523117, -43.5448380, -14.2503357, 14.3258343
41: -55.4188004, -32.9556274, -55.4518356, -32.9578323, -16.6201401, 16.7079659
42: -29.4492226, -9.8977280, -29.4704857, -9.8968563, -17.1825752, 17.2656250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=90, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=96, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5220173, upper bound: 12.5299726
time: 10.02 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664143, upper bound: 12.5342492
time: 13.52 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.1123810, 3.6617441, -12.1110039, 3.6669197, -13.8543854, 13.8594131
1: -3.6548982, 7.3930950, -3.6498826, 7.3944283, -8.4667549, 8.4852524
2: -0.7491778, 13.4309778, -0.7310567, 13.4304838, -13.4351807, 13.4260788
3: -1.1173092, 11.2957344, -1.1049546, 11.2967787, -12.0038033, 11.9826927
4: -11.0880947, 5.4623585, -11.0890055, 5.4797559, -14.6310577, 14.6298141
5: 1.8514986, 17.7415314, 1.8649597, 17.7404594, -15.8889608, 15.8765717
6: -39.8532600, -18.2886353, -39.8555145, -18.2522163, -15.0944786, 15.0597763
7: -3.5619993, 12.2460957, -3.5349736, 12.2527351, -13.5866928, 13.5695686
8: -6.7139902, 8.5643768, -6.7000284, 8.5668297, -12.0990372, 12.0714035
9: -4.7488341, 11.7016373, -4.7681227, 11.6872044, -12.9489365, 12.9738197
10: 1.3414726, 25.7528400, 1.3454571, 25.7194805, -20.8606873, 20.8869247
11: -11.4817314, 4.2910275, -11.4794416, 4.2850728, -15.7668037, 15.7704697
12: -11.8861513, 9.8441286, -11.8851032, 9.8212814, -14.9834938, 15.0008736
13: -18.5362625, 6.7141762, -18.5348320, 6.7065678, -16.5813293, 16.5560112
14: 4.9964657, 36.4334450, 5.0035477, 36.3849792, -26.6454773, 26.6928024
15: -8.6480989, 9.2111769, -8.6775112, 9.2232876, -17.8713875, 17.8886871
16: -16.7211609, 2.5577321, -16.7068481, 2.5434918, -14.7691422, 14.7975235
17: 6.2303452, 30.6567917, 6.2326875, 30.6347580, -17.1935158, 17.1711044
18: -14.3589716, 5.1092901, -14.3686581, 5.1135716, -14.3569965, 14.3710556
19: -20.2604599, -4.3374887, -20.2636280, -4.3331103, -14.5137901, 14.5112381
20: -2.4047534, 11.2054329, -2.3983939, 11.2082558, -12.5897141, 12.5816040
21: -11.0590954, 3.2585340, -11.0590420, 3.2483420, -14.3074379, 14.3175755
22: -3.6942964, 13.0960274, -3.6977363, 13.0756330, -14.9011879, 14.9068260
23: -14.5548859, 0.3012378, -14.5745430, 0.3146362, -14.2629700, 14.2777901
24: -19.9278889, -5.1284418, -19.9266968, -5.1300969, -9.2497482, 9.2460442
25: -5.4489503, 10.8525734, -5.4475613, 10.8341351, -13.7731743, 13.7691994
26: -20.9906998, 1.1921096, -21.0076466, 1.1630664, -19.2841644, 19.2808228
27: -16.0158291, 2.1631193, -15.9940186, 2.1600142, -13.1796722, 13.1840630
28: -12.7757225, 4.6015558, -12.7861462, 4.6114531, -17.3871765, 17.3877029
29: -5.5638876, 11.8726206, -5.5915046, 11.8544645, -14.8945427, 14.9040146
30: -10.0395679, 6.2212029, -10.0363579, 6.1997876, -13.5288734, 13.5366096
31: -10.9572830, 6.9486837, -10.9552679, 6.9484797, -14.6197510, 14.6186104
32: -24.8921375, -4.6021852, -24.8717155, -4.5935030, -13.2648392, 13.2220154
33: -69.3177338, -40.1334915, -69.2865982, -40.1215363, -16.6182747, 16.5977249
34: -53.7594109, -30.9297981, -53.7322998, -30.9249153, -14.1349411, 14.0811958
35: -47.8173103, -26.0822563, -47.8022766, -26.0759182, -13.0023689, 12.9769478
36: -42.8284912, -19.2925797, -42.8007622, -19.2932701, -15.0902824, 15.0505943
37: -86.6719131, -55.5705414, -86.6606522, -55.5773926, -18.8844986, 18.8849869
38: -52.9299583, -24.3647079, -52.9059448, -24.3455448, -18.3187256, 18.2832108
39: -76.5356827, -44.6470947, -76.5337067, -44.6340790, -16.0479889, 16.0382195
40: -67.2650909, -43.5294418, -67.2223740, -43.5328979, -14.2782860, 14.2751789
41: -55.4410400, -32.9637260, -55.4041443, -32.9724159, -16.6620941, 16.6438866
42: -29.4546795, -9.9160519, -29.4485874, -9.9134483, -17.2157555, 17.2269173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4544338
time: 39.57 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5663393, upper bound: 12.4786982
time: 8.50 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1135406, 3.6622057, -12.1185102, 3.6702261, -13.8578529, 13.8699074
1: -3.6536160, 7.3933134, -3.6507468, 7.3965693, -8.4714012, 8.4863548
2: -0.7512478, 13.4312077, -0.7372511, 13.4372501, -13.4478798, 13.4344444
3: -1.1258838, 11.2966175, -1.1225821, 11.3196716, -12.0353622, 11.9988995
4: -11.0882359, 5.4637070, -11.0926228, 5.4858637, -14.6372757, 14.6312408
5: 1.8433609, 17.7424145, 1.8489079, 17.7609024, -15.9175415, 15.8935070
6: -39.8628464, -18.2880192, -39.8766365, -18.2261963, -15.1368408, 15.0782318
7: -3.5755079, 12.2469912, -3.5644715, 12.2832651, -13.6309357, 13.5944443
8: -6.7134924, 8.5645733, -6.7048364, 8.5708122, -12.1181221, 12.0736771
9: -4.7491384, 11.7057228, -4.7774506, 11.6979704, -12.9581032, 12.9903412
10: 1.3402615, 25.7576809, 1.3273945, 25.7296791, -20.8699341, 20.9099121
11: -11.4876842, 4.2911267, -11.4988689, 4.3056021, -15.7932863, 15.7899952
12: -11.8881779, 9.8450165, -11.8898344, 9.8277569, -14.9797897, 15.0178032
13: -18.5406857, 6.7143054, -18.5443764, 6.7218971, -16.5912704, 16.5621223
14: 4.9830723, 36.4336395, 4.9755087, 36.3987885, -26.6722641, 26.7194214
15: -8.6483250, 9.2223501, -8.7113962, 9.2499199, -17.8982449, 17.9337463
16: -16.7195511, 2.5582194, -16.7132797, 2.5443010, -14.7821045, 14.8076630
17: 6.2192516, 30.6574078, 6.2103291, 30.6484203, -17.2176437, 17.1907234
18: -14.3593960, 5.1133051, -14.3848190, 5.1238661, -14.3661633, 14.3908081
19: -20.2614498, -4.3365636, -20.2746696, -4.3305840, -14.5201187, 14.5172768
20: -2.4103849, 11.2059479, -2.4130363, 11.2228708, -12.6115799, 12.5949783
21: -11.0598564, 3.2590547, -11.0664959, 3.2561202, -14.3159771, 14.3255501
22: -3.6943266, 13.0980406, -3.7107356, 13.0860434, -14.9162598, 14.9317131
23: -14.5562963, 0.3036609, -14.5877857, 0.3200669, -14.2703171, 14.2826157
24: -19.9284000, -5.1219535, -19.9421940, -5.1171827, -9.2587471, 9.2705536
25: -5.4493322, 10.8565855, -5.4611969, 10.8438301, -13.7812309, 13.7892723
26: -20.9914684, 1.2004611, -21.0340881, 1.1809289, -19.2946243, 19.3074646
27: -16.0159092, 2.1643777, -16.0011902, 2.1638632, -13.1850471, 13.1848640
28: -12.7768660, 4.6019964, -12.7947216, 4.6138282, -17.3906937, 17.3967171
29: -5.5649037, 11.8734732, -5.5962553, 11.8601809, -14.9036980, 14.9189987
30: -10.0431166, 6.2218094, -10.0475035, 6.2117462, -13.5456657, 13.5465240
31: -10.9583111, 6.9488997, -10.9644871, 6.9517603, -14.6322556, 14.6244125
32: -24.9021606, -4.6017485, -24.8919029, -4.5702095, -13.3013763, 13.2379837
33: -69.3178253, -40.1296387, -69.2985382, -40.1085281, -16.6288376, 16.6146469
34: -53.7595444, -30.9282494, -53.7373772, -30.9190712, -14.1415977, 14.0878868
35: -47.8177681, -26.0807304, -47.8075180, -26.0701256, -13.0069885, 12.9867172
36: -42.8291702, -19.2921143, -42.8023224, -19.2888222, -15.0958405, 15.0543747
37: -86.6735229, -55.5559464, -86.6915970, -55.5471878, -18.9093285, 18.9308739
38: -52.9305305, -24.3643112, -52.9083099, -24.3419514, -18.3238983, 18.2876053
39: -76.5363083, -44.6444626, -76.5460739, -44.6252365, -16.0553894, 16.0558510
40: -67.2660522, -43.5282097, -67.2295532, -43.5318069, -14.2878227, 14.2812748
41: -55.4416580, -32.9631424, -55.4093475, -32.9705963, -16.6659470, 16.6489029
42: -29.4541969, -9.9155731, -29.4519596, -9.9097996, -17.2228317, 17.2306786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=96, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4890084
time: 7.67 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5663393, upper bound: 12.5138163
time: 28.10 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1225967, 3.6604328, -12.1162291, 3.6662641, -13.8654060, 13.8612709
1: -3.6649683, 7.3932352, -3.6563382, 7.3962259, -8.4728889, 8.4817753
2: -0.7556199, 13.4378548, -0.7367823, 13.4323778, -13.4470406, 13.4414139
3: -1.1296616, 11.3277302, -1.1215229, 11.3080158, -12.0124397, 12.0290699
4: -11.0984936, 5.4685507, -11.0941315, 5.4823804, -14.6454315, 14.6421432
5: 1.8409982, 17.7660027, 1.8521643, 17.7478714, -15.9068737, 15.9138384
6: -39.8605499, -18.2412319, -39.8842239, -18.2281475, -15.0933266, 15.1376495
7: -3.5818975, 12.2752857, -3.5601482, 12.2574825, -13.6007156, 13.6176186
8: -6.7219563, 8.5686417, -6.7030354, 8.5681267, -12.1108360, 12.0941067
9: -4.7633996, 11.6899977, -4.7760305, 11.6971617, -12.9788742, 12.9680099
10: 1.3154888, 25.7414227, 1.3271222, 25.7378311, -20.9074478, 20.8840485
11: -11.4959450, 4.3061485, -11.4927683, 4.2851939, -15.7811394, 15.7989168
12: -11.8924770, 9.8462763, -11.8984823, 9.8271465, -15.0111389, 14.9961967
13: -18.5485802, 6.7213249, -18.5501938, 6.7112670, -16.6076927, 16.5587769
14: 4.9658661, 36.4102631, 4.9595261, 36.4000893, -26.6959991, 26.6991653
15: -8.6910658, 9.2272711, -8.6882219, 9.2479553, -17.9390221, 17.9154930
16: -16.7333755, 2.5461888, -16.7165756, 2.5415609, -14.7764015, 14.8045502
17: 6.2120318, 30.6642971, 6.1990361, 30.6439514, -17.1996460, 17.2156715
18: -14.3810081, 5.1177468, -14.3731070, 5.1234474, -14.3867531, 14.3759022
19: -20.2732410, -4.3332853, -20.2699242, -4.3272223, -14.5214615, 14.5275345
20: -2.4134266, 11.2302761, -2.4140384, 11.2231188, -12.6070824, 12.6200752
21: -11.0678205, 3.2665994, -11.0654602, 3.2503054, -14.3181257, 14.3320599
22: -3.7024345, 13.1037703, -3.6987073, 13.0843086, -14.9244804, 14.9214211
23: -14.5661335, 0.3084707, -14.5780268, 0.3231268, -14.2727280, 14.2932167
24: -19.9463272, -5.1200881, -19.9307480, -5.1180429, -9.2806206, 9.2519302
25: -5.4652958, 10.8572750, -5.4542818, 10.8448286, -13.7926331, 13.7792664
26: -21.0228195, 1.2025397, -21.0151138, 1.1812620, -19.3190079, 19.2918091
27: -16.0147781, 2.1730344, -16.0036812, 2.1742296, -13.1728020, 13.2058220
28: -12.7775869, 4.6078539, -12.7941322, 4.6209021, -17.3984890, 17.4019852
29: -5.5672312, 11.8757267, -5.5947208, 11.8599348, -14.9075623, 14.9208069
30: -10.0489683, 6.2271872, -10.0459175, 6.2027826, -13.5358810, 13.5513458
31: -10.9626293, 6.9540744, -10.9599524, 6.9505396, -14.6243362, 14.6369705
32: -24.8964348, -4.5540657, -24.9035416, -4.5667839, -13.2617645, 13.2949829
33: -69.2996826, -40.1208992, -69.2935944, -40.1107101, -16.6059990, 16.6039162
34: -53.7379456, -30.9167194, -53.7424240, -30.9127121, -14.1136742, 14.1090431
35: -47.8054123, -26.0756893, -47.8068619, -26.0697098, -12.9986229, 12.9879112
36: -42.8095284, -19.2801056, -42.8086472, -19.2807255, -15.0818863, 15.0723877
37: -86.6928482, -55.5492783, -86.6664963, -55.5494156, -18.9410019, 18.9006996
38: -52.9054108, -24.3487873, -52.9213028, -24.3288651, -18.3027458, 18.3187180
39: -76.5335007, -44.6405869, -76.5359268, -44.6295395, -16.0603561, 16.0446434
40: -67.2517929, -43.5238953, -67.2314072, -43.5238380, -14.2765350, 14.3005962
41: -55.4235001, -32.9527740, -55.4169731, -32.9588051, -16.6509476, 16.6747246
42: -29.4523869, -9.8874226, -29.4590263, -9.8887863, -17.2128296, 17.2591553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5636971, upper bound: 12.4660420
time: 13.32 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5674506, upper bound: 12.5091083
time: 6.95 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1210442, 3.6606712, -12.1149397, 3.6717038, -13.8703194, 13.8612137
1: -3.6617301, 7.3935337, -3.6519370, 7.3998489, -8.4844856, 8.4839363
2: -0.7543463, 13.4380836, -0.7357163, 13.4343376, -13.4490356, 13.4420586
3: -1.1296389, 11.3246088, -1.1216152, 11.3056335, -12.0142365, 12.0350552
4: -11.0947723, 5.4688249, -11.0892220, 5.4833765, -14.6454544, 14.6399765
5: 1.8410058, 17.7648926, 1.8517270, 17.7477760, -15.9067707, 15.9131660
6: -39.8606873, -18.2460747, -39.8915100, -18.2345905, -15.0928040, 15.1419525
7: -3.5813255, 12.2753258, -3.5616331, 12.2607231, -13.6102676, 13.6184883
8: -6.7201862, 8.5688019, -6.7025785, 8.5692158, -12.1125336, 12.0949535
9: -4.7654028, 11.6899567, -4.7838326, 11.7174349, -12.9928627, 12.9724121
10: 1.3112583, 25.7413177, 1.3130393, 25.7570553, -20.9271622, 20.8957901
11: -11.4976187, 4.3055577, -11.5001068, 4.2911325, -15.7887516, 15.8056641
12: -11.8926802, 9.8459778, -11.9041538, 9.8321438, -15.0150299, 15.0011864
13: -18.5489349, 6.7212601, -18.5528469, 6.7219000, -16.6116409, 16.5607185
14: 4.9607248, 36.4103279, 4.9411497, 36.4380836, -26.7315521, 26.7143631
15: -8.6883411, 9.2277489, -8.6854439, 9.2522564, -17.9405975, 17.9131927
16: -16.7350349, 2.5462620, -16.7270813, 2.5544178, -14.7952843, 14.8118324
17: 6.2111254, 30.6643143, 6.1906176, 30.6505985, -17.1976013, 17.2293205
18: -14.3772030, 5.1180453, -14.3693829, 5.1251197, -14.3891106, 14.3788376
19: -20.2719765, -4.3330398, -20.2714996, -4.3260779, -14.5211792, 14.5331802
20: -2.4136486, 11.2291393, -2.4185963, 11.2220573, -12.6068077, 12.6214828
21: -11.0660419, 3.2665269, -11.0675106, 3.2509418, -14.3169842, 14.3340378
22: -3.7023044, 13.1042604, -3.7052553, 13.0885315, -14.9279327, 14.9335709
23: -14.5665054, 0.3084753, -14.5825148, 0.3244112, -14.2747955, 14.2957344
24: -19.9447823, -5.1201391, -19.9294376, -5.1176777, -9.2820358, 9.2535057
25: -5.4649363, 10.8572960, -5.4565563, 10.8463116, -13.7920761, 13.7866402
26: -21.0196495, 1.2026272, -21.0140915, 1.1833529, -19.3198204, 19.2955704
27: -16.0146542, 2.1740694, -16.0135384, 2.1777129, -13.1766205, 13.2073326
28: -12.7777653, 4.6086879, -12.8024416, 4.6242666, -17.4020309, 17.4111290
29: -5.5673280, 11.8754272, -5.5970020, 11.8615532, -14.9084358, 14.9281158
30: -10.0498819, 6.2269192, -10.0511236, 6.2092772, -13.5439301, 13.5559921
31: -10.9624233, 6.9546523, -10.9676476, 6.9525981, -14.6258163, 14.6435814
32: -24.8965225, -4.5605950, -24.9134560, -4.5755978, -13.2615738, 13.3049088
33: -69.2993851, -40.1165009, -69.3252258, -40.0974731, -16.6146851, 16.6373444
34: -53.7379074, -30.9136086, -53.7698174, -30.9017315, -14.1222153, 14.1355629
35: -47.8053360, -26.0739155, -47.8244095, -26.0622082, -13.0055618, 13.0074425
36: -42.8094254, -19.2787132, -42.8291702, -19.2745266, -15.0857239, 15.0888634
37: -86.6925735, -55.5476532, -86.6772003, -55.5439148, -18.9453316, 18.9079666
38: -52.9053802, -24.3468266, -52.9478836, -24.3211479, -18.3075333, 18.3339996
39: -76.5332642, -44.6382599, -76.5524597, -44.6226959, -16.0657005, 16.0608330
40: -67.2516327, -43.5209885, -67.2524490, -43.5168381, -14.2809982, 14.2990532
41: -55.4235420, -32.9502029, -55.4407921, -32.9505844, -16.6539917, 16.6867561
42: -29.4526100, -9.8955307, -29.4652557, -9.9009027, -17.2108269, 17.2631645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5234525, upper bound: 12.5470845
time: 10.62 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5674503, upper bound: 12.5508635
time: 7.01 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.1402817, 3.6745524, -12.1078787, 3.6597161, -13.8930130, 13.8629189
1: -3.6719937, 7.3931217, -3.6543617, 7.3781962, -8.4730568, 8.4858685
2: -0.7651426, 13.4301510, -0.7292621, 13.4110031, -13.4465714, 13.4323387
3: -1.1425248, 11.3182144, -1.1282203, 11.2863121, -12.0284386, 12.0181656
4: -11.1209755, 5.4777288, -11.0977192, 5.4634752, -14.6459274, 14.6525993
5: 1.8247480, 17.7589302, 1.8446426, 17.7317963, -15.9070482, 15.9127197
6: -39.9386406, -18.2195778, -39.9190712, -18.2474060, -15.0898628, 15.2005196
7: -3.6162395, 12.2764893, -3.5743873, 12.2299442, -13.6068726, 13.6164398
8: -6.7224865, 8.5684013, -6.6939774, 8.5569668, -12.0929451, 12.1080513
9: -4.7893052, 11.7395611, -4.7724447, 11.7028599, -13.0047112, 12.9820023
10: 1.2969699, 25.7616005, 1.3311076, 25.7180977, -20.9047470, 20.9156647
11: -11.5084572, 4.3129659, -11.4931355, 4.2843289, -15.7927856, 15.8061008
12: -11.8893681, 9.8658323, -11.8672752, 9.8137865, -14.9833870, 15.0079079
13: -18.5431671, 6.7575169, -18.5375004, 6.7252169, -16.6201324, 16.5429268
14: 4.9661264, 36.4832497, 5.0125017, 36.4097900, -26.7145081, 26.6948013
15: -8.7216625, 9.2945738, -8.6883764, 9.3008480, -18.0225105, 17.9829502
16: -16.7520123, 2.5363321, -16.7175293, 2.5096107, -14.7792206, 14.8170052
17: 6.2231355, 30.6953888, 6.2659860, 30.6393852, -17.2050934, 17.1857910
18: -14.4085064, 5.1274571, -14.3845844, 5.1174130, -14.4103146, 14.4054508
19: -20.2823601, -4.3237200, -20.2605247, -4.3265514, -14.5317307, 14.5245514
20: -2.4242988, 11.2303333, -2.4009781, 11.2057552, -12.5978546, 12.6135674
21: -11.0756054, 3.2701859, -11.0563307, 3.2423964, -14.3180017, 14.3265171
22: -3.7007201, 13.1428986, -3.6639271, 13.0931149, -14.9340363, 14.9335480
23: -14.5921230, 0.3472507, -14.5691986, 0.3440733, -14.3056145, 14.2988129
24: -19.9479218, -5.1113234, -19.9286041, -5.1081147, -9.2710800, 9.2616463
25: -5.4636164, 10.8853989, -5.4312410, 10.8593416, -13.7965927, 13.7831497
26: -21.0096416, 1.2552996, -20.9583759, 1.1857159, -19.3052139, 19.2924118
27: -16.0388870, 2.1760068, -15.9985199, 2.1659849, -13.1892128, 13.2075233
28: -12.8064842, 4.6447282, -12.7770758, 4.6334791, -17.4399643, 17.4218044
29: -5.5759163, 11.9235039, -5.5498419, 11.8664474, -14.9267693, 14.9251213
30: -10.0468712, 6.2381783, -10.0293179, 6.1960640, -13.5367279, 13.5611076
31: -10.9954624, 6.9530592, -10.9692917, 6.9495387, -14.6355209, 14.6446838
32: -24.9374332, -4.5598440, -24.9180927, -4.5890226, -13.2536240, 13.3095093
33: -69.3622742, -40.1039696, -69.2961578, -40.1318970, -16.6423798, 16.6401062
34: -53.7974739, -30.9119186, -53.7507706, -30.9339905, -14.1066780, 14.1432838
35: -47.8295059, -26.0686302, -47.8046188, -26.0705967, -12.9961357, 12.9875374
36: -42.8163376, -19.2845707, -42.7833519, -19.2973633, -15.0792313, 15.0606155
37: -86.7059174, -55.5476875, -86.6605225, -55.5600777, -18.9156723, 18.9148865
38: -52.9683685, -24.3295918, -52.9188881, -24.3418770, -18.3219109, 18.3307495
39: -76.5753632, -44.6282883, -76.5453339, -44.6408424, -16.0814209, 16.0799904
40: -67.3095398, -43.5384064, -67.2378845, -43.5573730, -14.2878838, 14.3498249
41: -55.4684944, -32.9600296, -55.4241371, -32.9784431, -16.6433640, 16.6861229
42: -29.4699879, -9.9096870, -29.4585876, -9.9103003, -17.2035675, 17.2523880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5619689, upper bound: 12.4862491
time: 6.70 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5652797, upper bound: 12.5124337
time: 9.42 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1438370, 3.6702995, -12.1110916, 3.6582615, -13.8937759, 13.8610458
1: -3.6813555, 7.3912868, -3.6619415, 7.3796034, -8.4781036, 8.4777546
2: -0.7675083, 13.4303923, -0.7308072, 13.4124937, -13.4526215, 13.4323959
3: -1.1434126, 11.3279772, -1.1300923, 11.2959890, -12.0264130, 12.0274029
4: -11.1278858, 5.4792223, -11.1025572, 5.4633069, -14.6579056, 14.6596107
5: 1.8243275, 17.7636261, 1.8458743, 17.7376328, -15.9133053, 15.9177513
6: -39.9328423, -18.1977444, -39.9301682, -18.2243862, -15.0789909, 15.2273445
7: -3.6163292, 12.2758312, -3.5763931, 12.2331104, -13.6050491, 13.6112518
8: -6.7260995, 8.5688267, -6.6970181, 8.5579491, -12.1055984, 12.1085682
9: -4.7947874, 11.7200928, -4.7797890, 11.7057638, -13.0146484, 12.9705696
10: 1.2880359, 25.7434158, 1.3149447, 25.7281952, -20.9259796, 20.9060364
11: -11.5089331, 4.3076739, -11.4947548, 4.2842422, -15.7931747, 15.8024292
12: -11.8927660, 9.8621378, -11.8767672, 9.8181667, -14.9930305, 15.0080109
13: -18.5490170, 6.7495337, -18.5453014, 6.7296000, -16.6422119, 16.5339851
14: 4.9537973, 36.4464340, 4.9917059, 36.4245224, -26.7480011, 26.6647797
15: -8.7309361, 9.2937851, -8.6986713, 9.3046141, -18.0355492, 17.9924564
16: -16.7588158, 2.5243242, -16.7278290, 2.5068579, -14.7780838, 14.8093224
17: 6.2192011, 30.6896400, 6.2513981, 30.6475544, -17.1994057, 17.1984901
18: -14.4149132, 5.1289196, -14.3880615, 5.1199374, -14.4178886, 14.4035969
19: -20.2847672, -4.3210983, -20.2651596, -4.3225236, -14.5338745, 14.5340271
20: -2.4224610, 11.2409410, -2.4069216, 11.2196932, -12.6055107, 12.6265106
21: -11.0786943, 3.2708664, -11.0601511, 3.2434518, -14.3221464, 14.3310175
22: -3.6959648, 13.1444960, -3.6648033, 13.0955515, -14.9266777, 14.9389038
23: -14.5911608, 0.3509867, -14.5702114, 0.3481934, -14.3113976, 14.3059959
24: -19.9512596, -5.1108193, -19.9317532, -5.1076274, -9.2728462, 9.2630997
25: -5.4666491, 10.8849087, -5.4372826, 10.8614864, -13.7921600, 13.7889977
26: -21.0159359, 1.2551551, -20.9644699, 1.1882660, -19.3086052, 19.2977829
27: -16.0307713, 2.1831787, -16.0079803, 2.1777925, -13.1809883, 13.2244759
28: -12.8006287, 4.6494703, -12.7830782, 4.6416287, -17.4422569, 17.4325485
29: -5.5752749, 11.9236994, -5.5513358, 11.8682117, -14.9217682, 14.9357719
30: -10.0476685, 6.2326403, -10.0327702, 6.1980553, -13.5359116, 13.5569572
31: -10.9924412, 6.9553037, -10.9721317, 6.9512177, -14.6357689, 14.6491013
32: -24.9292450, -4.5347290, -24.9321175, -4.5630198, -13.2420120, 13.3384819
33: -69.3323517, -40.1010361, -69.3029022, -40.1282654, -16.6098709, 16.6391602
34: -53.7710037, -30.9032936, -53.7607155, -30.9248276, -14.0774307, 14.1657372
35: -47.8126488, -26.0661392, -47.8085136, -26.0676346, -12.9813499, 12.9951782
36: -42.7962952, -19.2756577, -42.7900543, -19.2861462, -15.0664825, 15.0774574
37: -86.6970673, -55.5461044, -86.6636047, -55.5572815, -18.9172516, 18.9147530
38: -52.9428635, -24.3169117, -52.9322662, -24.3259811, -18.3028069, 18.3598938
39: -76.5611572, -44.6280365, -76.5465775, -44.6416054, -16.0741386, 16.0810394
40: -67.2899094, -43.5330429, -67.2451172, -43.5504379, -14.2804680, 14.3652859
41: -55.4462852, -32.9504089, -55.4358673, -32.9659348, -16.6272087, 16.7130661
42: -29.4655075, -9.8843441, -29.4683304, -9.8864613, -17.1975594, 17.2769051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5623429, upper bound: 12.4642308
time: 10.36 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665427, upper bound: 12.5092051
time: 11.83 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1422930, 3.6705279, -12.1098003, 3.6636946, -13.8986816, 13.8610306
1: -3.6781065, 7.3915720, -3.6575303, 7.3832173, -8.4897003, 8.4799213
2: -0.7662381, 13.4306192, -0.7297215, 13.4144859, -13.4545975, 13.4330177
3: -1.1433887, 11.3248615, -1.1301872, 11.2936153, -12.0282364, 12.0333824
4: -11.1241493, 5.4795046, -11.0976534, 5.4643259, -14.6578827, 14.6574364
5: 1.8243642, 17.7625198, 1.8454552, 17.7375908, -15.9132271, 15.9170647
6: -39.9329605, -18.2025471, -39.9374390, -18.2308407, -15.0784645, 15.2315903
7: -3.6157708, 12.2758732, -3.5778844, 12.2363605, -13.6145706, 13.6121178
8: -6.7243428, 8.5689869, -6.6965790, 8.5590248, -12.1072693, 12.1093979
9: -4.7967978, 11.7200613, -4.7876225, 11.7260580, -13.0286598, 12.9749680
10: 1.2837963, 25.7433167, 1.3009191, 25.7474213, -20.9457016, 20.9177246
11: -11.5106144, 4.3070493, -11.5020885, 4.2901926, -15.8008070, 15.8091373
12: -11.8929291, 9.8618202, -11.8824806, 9.8231544, -14.9969139, 15.0130005
13: -18.5494175, 6.7494230, -18.5479412, 6.7402420, -16.6461601, 16.5359535
14: 4.9486580, 36.4464836, 4.9733238, 36.4625359, -26.7835846, 26.6800308
15: -8.7282047, 9.2942724, -8.6959038, 9.3089161, -18.0371208, 17.9901772
16: -16.7604523, 2.5244322, -16.7383652, 2.5197473, -14.7969055, 14.8165741
17: 6.2182751, 30.6896744, 6.2429667, 30.6541729, -17.1973648, 17.2121277
18: -14.4111013, 5.1292248, -14.3843460, 5.1216326, -14.4202633, 14.4065170
19: -20.2834873, -4.3208661, -20.2667198, -4.3213439, -14.5336151, 14.5396690
20: -2.4226582, 11.2398167, -2.4114723, 11.2186213, -12.6052094, 12.6279259
21: -11.0769272, 3.2707922, -11.0622139, 3.2440877, -14.3210144, 14.3330059
22: -3.6958175, 13.1449909, -3.6713252, 13.0997925, -14.9301071, 14.9510498
23: -14.5915394, 0.3509891, -14.5747147, 0.3494666, -14.3135033, 14.3085136
24: -19.9497337, -5.1108646, -19.9304352, -5.1072440, -9.2742653, 9.2646561
25: -5.4662924, 10.8849316, -5.4395494, 10.8629761, -13.7915688, 13.7963753
26: -21.0127621, 1.2552719, -20.9635124, 1.1903737, -19.3094521, 19.3015213
27: -16.0306454, 2.1842299, -16.0178719, 2.1812901, -13.1848183, 13.2259827
28: -12.8007755, 4.6503096, -12.7914028, 4.6449909, -17.4457664, 17.4417114
29: -5.5753202, 11.9233885, -5.5536337, 11.8698330, -14.9226379, 14.9431114
30: -10.0485697, 6.2323589, -10.0379457, 6.2045231, -13.5439262, 13.5616035
31: -10.9922075, 6.9558587, -10.9798365, 6.9533205, -14.6372223, 14.6556969
32: -24.9293518, -4.5412416, -24.9420700, -4.5718336, -13.2418098, 13.3483925
33: -69.3320312, -40.0966034, -69.3345795, -40.1149521, -16.6185455, 16.6725883
34: -53.7710114, -30.9001789, -53.7880783, -30.9138126, -14.0859680, 14.1922760
35: -47.8125229, -26.0643768, -47.8260574, -26.0601120, -12.9883041, 13.0147400
36: -42.7961655, -19.2742462, -42.8105392, -19.2799797, -15.0703239, 15.0939751
37: -86.6967850, -55.5444832, -86.6742706, -55.5518036, -18.9215851, 18.9220123
38: -52.9428101, -24.3149452, -52.9588623, -24.3182068, -18.3075867, 18.3751678
39: -76.5609741, -44.6257133, -76.5631104, -44.6347809, -16.0794830, 16.0972214
40: -67.2896957, -43.5301514, -67.2661362, -43.5433807, -14.2849274, 14.3637238
41: -55.4463272, -32.9478531, -55.4596672, -32.9576836, -16.6302185, 16.7251129
42: -29.4656944, -9.8924541, -29.4745579, -9.8985939, -17.1955605, 17.2809143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5221770, upper bound: 12.5465501
time: 26.11 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5665424, upper bound: 12.5506992
time: 6.80 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.1420746, 3.6874330, -12.1183672, 3.6808615, -13.9012642, 13.8880882
1: -3.6722989, 7.4053278, -3.6585968, 7.3981323, -8.4782104, 8.5059414
2: -0.7663668, 13.4443054, -0.7416826, 13.4348679, -13.4594803, 13.4587860
3: -1.1433160, 11.3281803, -1.1279161, 11.3032665, -12.0423279, 12.0274792
4: -11.1216192, 5.4915318, -11.1064720, 5.4868550, -14.6582184, 14.6751671
5: 1.8234053, 17.7662773, 1.8410063, 17.7443867, -15.9209814, 15.9252710
6: -39.9404335, -18.2211304, -39.9157562, -18.2507877, -15.1088905, 15.1960449
7: -3.6169207, 12.2926502, -3.5769200, 12.2568636, -13.6167450, 13.6516151
8: -6.7231998, 8.5777159, -6.7009625, 8.5722885, -12.1106186, 12.1037750
9: -4.7905149, 11.7460299, -4.7749643, 11.7143059, -13.0206223, 12.9967690
10: 1.2948942, 25.7695179, 1.3277717, 25.7326241, -20.9211349, 20.9445953
11: -11.5117111, 4.3136787, -11.4980965, 4.2868857, -15.7985973, 15.8117752
12: -11.9083099, 9.8671284, -11.8990002, 9.8247490, -15.0134392, 15.0238228
13: -18.5472126, 6.7585735, -18.5437069, 6.7215910, -16.5963326, 16.5969467
14: 4.9453068, 36.4837799, 4.9767103, 36.4056091, -26.7160797, 26.7432709
15: -8.7179031, 9.2954073, -8.6805401, 9.2825260, -18.0004292, 17.9759483
16: -16.7533360, 2.5581434, -16.7188911, 2.5454721, -14.8141251, 14.8244324
17: 6.1909604, 30.6960068, 6.2115235, 30.6495266, -17.2453651, 17.2299194
18: -14.4119062, 5.1323714, -14.3906536, 5.1259542, -14.4207458, 14.4172001
19: -20.2897301, -4.3235469, -20.2733784, -4.3239660, -14.5482025, 14.5385590
20: -2.4347968, 11.2315331, -2.4185398, 11.2121487, -12.6139641, 12.6254158
21: -11.0850391, 3.2712810, -11.0726643, 3.2501464, -14.3351860, 14.3439455
22: -3.7231863, 13.1434021, -3.7008212, 13.1051798, -14.9755821, 14.9498634
23: -14.5990925, 0.3475714, -14.5819826, 0.3440251, -14.3220749, 14.3112411
24: -19.9479580, -5.1106930, -19.9293613, -5.1134844, -9.2818031, 9.2683678
25: -5.4751196, 10.8859119, -5.4511089, 10.8573771, -13.8247375, 13.7966309
26: -21.0416718, 1.2562582, -21.0120831, 1.2094357, -19.3627853, 19.3050385
27: -16.0401745, 2.1776795, -15.9998999, 2.1688240, -13.2158737, 13.2043762
28: -12.8149652, 4.6451364, -12.7926254, 4.6372008, -17.4521656, 17.4377613
29: -5.6032228, 11.9238453, -5.5955992, 11.8857765, -14.9745865, 14.9412689
30: -10.0573158, 6.2394094, -10.0465136, 6.2054653, -13.5566711, 13.5691872
31: -10.9973736, 6.9532652, -10.9733200, 6.9495344, -14.6563492, 14.6482391
32: -24.9386444, -4.5618553, -24.9086533, -4.5921588, -13.2743187, 13.2979355
33: -69.3628998, -40.0905228, -69.3060226, -40.1086807, -16.6504822, 16.6559944
34: -53.7983093, -30.9033051, -53.7527924, -30.9188061, -14.1392441, 14.1276283
35: -47.8306808, -26.0673351, -47.8076401, -26.0702095, -13.0114059, 12.9971008
36: -42.8292999, -19.2836685, -42.8059692, -19.2903633, -15.1037254, 15.0663795
37: -86.7099457, -55.5392265, -86.6686859, -55.5462265, -18.9463310, 18.9282455
38: -52.9767036, -24.3288555, -52.9334030, -24.3423634, -18.3360901, 18.3479309
39: -76.5760269, -44.6190186, -76.5500107, -44.6252823, -16.0997391, 16.0859604
40: -67.3108597, -43.5217743, -67.2433243, -43.5303154, -14.3223991, 14.3191814
41: -55.4691315, -32.9553452, -55.4171791, -32.9704056, -16.6760406, 16.6660538
42: -29.4706745, -9.9103279, -29.4560432, -9.9115372, -17.2357941, 17.2459641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5635137, upper bound: 12.5049642
time: 8.57 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664667, upper bound: 12.5302199
time: 18.26 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1456089, 3.6831632, -12.1215687, 3.6794009, -13.9020081, 13.8862152
1: -3.6816502, 7.4034643, -3.6661801, 7.3995223, -8.4832573, 8.4978352
2: -0.7687516, 13.4445744, -0.7432479, 13.4363565, -13.4655380, 13.4588623
3: -1.1442063, 11.3379288, -1.1297845, 11.3129377, -12.0403214, 12.0367317
4: -11.1285648, 5.4930196, -11.1113224, 5.4867196, -14.6702423, 14.6821709
5: 1.8229942, 17.7709560, 1.8422437, 17.7502441, -15.9267731, 15.9287128
6: -39.9346237, -18.1992702, -39.9268646, -18.2277870, -15.0980186, 15.2228584
7: -3.6170042, 12.2919445, -3.5789323, 12.2600250, -13.6149368, 13.6464462
8: -6.7268181, 8.5781116, -6.7040052, 8.5732651, -12.1232834, 12.1042862
9: -4.7959995, 11.7265730, -4.7822924, 11.7172184, -13.0305443, 12.9853439
10: 1.2860012, 25.7513103, 1.3115969, 25.7427158, -20.9423904, 20.9349976
11: -11.5122004, 4.3083825, -11.4997215, 4.2868133, -15.7990131, 15.8081036
12: -11.9117165, 9.8634481, -11.9085159, 9.8291283, -15.0230865, 15.0239105
13: -18.5530758, 6.7505913, -18.5515099, 6.7259421, -16.6183739, 16.5880241
14: 4.9329567, 36.4469681, 4.9558945, 36.4203720, -26.7495728, 26.7132492
15: -8.7271624, 9.2946291, -8.6907883, 9.2863102, -18.0134735, 17.9854164
16: -16.7601433, 2.5461280, -16.7292404, 2.5427511, -14.8130493, 14.8167839
17: 6.1870327, 30.6902714, 6.1969337, 30.6576691, -17.2396774, 17.2426033
18: -14.4182882, 5.1338415, -14.3941498, 5.1284876, -14.4283180, 14.4153442
19: -20.2921314, -4.3209333, -20.2780075, -4.3199391, -14.5503311, 14.5480309
20: -2.4329669, 11.2421494, -2.4244730, 11.2260799, -12.6216049, 12.6383629
21: -11.0881433, 3.2719541, -11.0764914, 3.2512031, -14.3393459, 14.3484459
22: -3.7184086, 13.1449795, -3.7016785, 13.1076107, -14.9682007, 14.9551582
23: -14.5981255, 0.3513165, -14.5829372, 0.3481269, -14.3278732, 14.3183899
24: -19.9513054, -5.1101809, -19.9325218, -5.1130228, -9.2835693, 9.2698555
25: -5.4781671, 10.8854151, -5.4571552, 10.8595495, -13.8202400, 13.8024673
26: -21.0479660, 1.2561226, -21.0182190, 1.2120008, -19.3661652, 19.3104248
27: -16.0320511, 2.1848512, -16.0093784, 2.1806316, -13.2076607, 13.2213440
28: -12.8090744, 4.6498876, -12.7985983, 4.6453762, -17.4544506, 17.4484863
29: -5.6025257, 11.9240398, -5.5971026, 11.8875656, -14.9695816, 14.9519234
30: -10.0581007, 6.2338805, -10.0499916, 6.2074103, -13.5558548, 13.5650444
31: -10.9943724, 6.9555097, -10.9761753, 6.9512119, -14.6565781, 14.6526413
32: -24.9304543, -4.5367613, -24.9226875, -4.5661392, -13.2627106, 13.3269081
33: -69.3329773, -40.0875931, -69.3128204, -40.1050186, -16.6178741, 16.6550331
34: -53.7718468, -30.8946152, -53.7627220, -30.9096298, -14.1100082, 14.1500626
35: -47.8138390, -26.0648479, -47.8115196, -26.0672798, -12.9966240, 13.0047684
36: -42.8092346, -19.2747383, -42.8126640, -19.2791748, -15.0909348, 15.0832176
37: -86.7011108, -55.5376549, -86.6717224, -55.5433807, -18.9479637, 18.9281006
38: -52.9512100, -24.3161430, -52.9467010, -24.3264656, -18.3169441, 18.3770676
39: -76.5618820, -44.6187363, -76.5512390, -44.6259918, -16.0924721, 16.0870132
40: -67.2912216, -43.5164032, -67.2505646, -43.5234261, -14.3149872, 14.3346310
41: -55.4469376, -32.9457016, -55.4289322, -32.9578896, -16.6598663, 16.6930122
42: -29.4661732, -9.8849792, -29.4657841, -9.8877363, -17.2298164, 17.2704239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5235955, upper bound: 12.5221107
time: 7.33 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5258383
time: 10.32 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -12.1365566, 3.6801229, -12.1190834, 3.6843829, -13.8964195, 13.8826904
1: -3.6775384, 7.4016309, -3.6630542, 7.4029160, -8.4937286, 8.4953346
2: -0.7612956, 13.4380322, -0.7401206, 13.4381304, -13.4591866, 13.4467621
3: -1.1265494, 11.3119106, -1.1213093, 11.3096857, -12.0259094, 12.0111275
4: -11.1212177, 5.4871898, -11.1062832, 5.4864082, -14.6687775, 14.6737709
5: 1.8390279, 17.7493763, 1.8499775, 17.7492962, -15.9102688, 15.8993988
6: -39.9136276, -18.2301407, -39.9245682, -18.2348423, -15.0790710, 15.1847916
7: -3.5869324, 12.2614498, -3.5669262, 12.2623405, -13.5995483, 13.6030693
8: -6.7202449, 8.5743332, -6.7040691, 8.5741377, -12.1226997, 12.0860271
9: -4.7886896, 11.7157621, -4.7897816, 11.7333984, -13.0280304, 12.9805489
10: 1.2998519, 25.7410393, 1.2987838, 25.7571068, -20.9391098, 20.9374771
11: -11.4944000, 4.2872415, -11.5010939, 4.2926617, -15.7870617, 15.7883358
12: -11.9071493, 9.8566628, -11.9121513, 9.8332767, -15.0100250, 15.0325851
13: -18.5438957, 6.7351651, -18.5497379, 6.7363777, -16.6161880, 16.5799866
14: 4.9558935, 36.4331589, 4.9509153, 36.4581947, -26.7585373, 26.7017059
15: -8.6905308, 9.2684460, -8.6878014, 9.2794418, -17.9699726, 17.9562473
16: -16.7553158, 2.5454469, -16.7413483, 2.5551291, -14.8218040, 14.8110847
17: 6.2084522, 30.6766109, 6.1995654, 30.6637344, -17.2180367, 17.2321167
18: -14.3983202, 5.1238656, -14.3900156, 5.1261392, -14.4109535, 14.4091225
19: -20.2798080, -4.3231688, -20.2785988, -4.3197269, -14.5440140, 14.5473137
20: -2.4185216, 11.2264166, -2.4233809, 11.2244835, -12.6079330, 12.6178970
21: -11.0789413, 3.2641287, -11.0777988, 3.2513075, -14.3302488, 14.3419275
22: -3.7052324, 13.1351290, -3.7081845, 13.1098557, -14.9467316, 14.9522591
23: -14.5853062, 0.3459048, -14.5860291, 0.3469841, -14.3251801, 14.3135223
24: -19.9342842, -5.1231394, -19.9307175, -5.1191568, -9.2604485, 9.2624283
25: -5.4641271, 10.8757677, -5.4590597, 10.8570070, -13.7995605, 13.8017845
26: -21.0183792, 1.2383614, -21.0165176, 1.2057617, -19.3403091, 19.3037109
27: -16.0247383, 2.1820264, -16.0191994, 2.1828794, -13.2106972, 13.2174721
28: -12.8006802, 4.6483307, -12.8057671, 4.6482649, -17.4489441, 17.4540977
29: -5.5978189, 11.9180155, -5.5983810, 11.8883257, -14.9554672, 14.9501190
30: -10.0478745, 6.2216415, -10.0516272, 6.2132826, -13.5539703, 13.5529251
31: -10.9849396, 6.9528074, -10.9828587, 6.9530764, -14.6522598, 14.6467743
32: -24.9103432, -4.5665398, -24.9225655, -4.5753555, -13.2465477, 13.3002777
33: -69.3206787, -40.0962067, -69.3443069, -40.0955429, -16.6096725, 16.6778831
34: -53.7667313, -30.8973694, -53.7900085, -30.9002476, -14.1118393, 14.1699867
35: -47.8084831, -26.0688057, -47.8286362, -26.0612545, -12.9938164, 13.0196838
36: -42.8075714, -19.2777901, -42.8324547, -19.2734318, -15.0910187, 15.0941429
37: -86.6699142, -55.5662193, -86.6808014, -55.5525627, -18.9063797, 18.9105301
38: -52.9487915, -24.3177528, -52.9727440, -24.3191757, -18.3173676, 18.3871613
39: -76.5493011, -44.6252289, -76.5671768, -44.6218185, -16.0801849, 16.0957642
40: -67.2838669, -43.5145683, -67.2706604, -43.5175972, -14.3133583, 14.3235645
41: -55.4418373, -32.9450073, -55.4521179, -32.9502411, -16.6579018, 16.7012215
42: -29.4630051, -9.8967285, -29.4724960, -9.9003181, -17.2240219, 17.2673721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4893752, upper bound: 12.5638365
time: 30.47 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5323473, upper bound: 12.5675748
time: 6.49 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1440754, 3.6834261, -12.1202450, 3.6848521, -13.9069252, 13.8861847
1: -3.6784005, 7.4037638, -3.6617696, 7.4031415, -8.4948463, 8.5000000
2: -0.7674818, 13.4447861, -0.7421657, 13.4383335, -13.4675369, 13.4594955
3: -1.1441662, 11.3348312, -1.1298699, 11.3105640, -12.0421143, 12.0427017
4: -11.1248331, 5.4933171, -11.1064234, 5.4877181, -14.6702271, 14.6800270
5: 1.8230257, 17.7698288, 1.8418155, 17.7501640, -15.9271383, 15.9280128
6: -39.9347610, -18.2041130, -39.9341393, -18.2342434, -15.0975075, 15.2271385
7: -3.6164021, 12.2919960, -3.5804195, 12.2632742, -13.6244812, 13.6472931
8: -6.7250576, 8.5783243, -6.7035408, 8.5743141, -12.1249771, 12.1051159
9: -4.7980027, 11.7265272, -4.7901120, 11.7374735, -13.0445404, 12.9897308
10: 1.2817435, 25.7512436, 1.2975702, 25.7619286, -20.9621124, 20.9466934
11: -11.5138702, 4.3077607, -11.5070915, 4.2927494, -15.8066196, 15.8148518
12: -11.9118891, 9.8631487, -11.9141884, 9.8341103, -15.0269623, 15.0289040
13: -18.5534630, 6.7505531, -18.5541477, 6.7365832, -16.6223373, 16.5899353
14: 4.9278164, 36.4469986, 4.9375153, 36.4584351, -26.7851715, 26.7285156
15: -8.7244453, 9.2951059, -8.6880617, 9.2905760, -18.0150223, 17.9831676
16: -16.7617741, 2.5462408, -16.7397594, 2.5556126, -14.8319283, 14.8240662
17: 6.1861076, 30.6902924, 6.1885285, 30.6643238, -17.2376404, 17.2562447
18: -14.4144640, 5.1341476, -14.3904514, 5.1301794, -14.4306755, 14.4182739
19: -20.2908592, -4.3206863, -20.2795887, -4.3187661, -14.5500412, 14.5536499
20: -2.4331446, 11.2410107, -2.4290178, 11.2250118, -12.6213112, 12.6397743
21: -11.0863705, 3.2718854, -11.0785618, 3.2518332, -14.3382034, 14.3504467
22: -3.7182732, 13.1455040, -3.7082281, 13.1118393, -14.9716454, 14.9673080
23: -14.5985355, 0.3513274, -14.5874290, 0.3494005, -14.3299789, 14.3209190
24: -19.9497776, -5.1102180, -19.9312019, -5.1126451, -9.2849884, 9.2714043
25: -5.4778004, 10.8854513, -5.4594297, 10.8610153, -13.8196716, 13.8098297
26: -21.0447769, 1.2562428, -21.0172348, 1.2141221, -19.3669739, 19.3141708
27: -16.0319138, 2.1859031, -16.0192738, 2.1841311, -13.2114716, 13.2228432
28: -12.8092422, 4.6506824, -12.8069181, 4.6487083, -17.4579506, 17.4575996
29: -5.6025858, 11.9237385, -5.5994053, 11.8891850, -14.9704704, 14.9592628
30: -10.0590277, 6.2336111, -10.0551891, 6.2138877, -13.5638847, 13.5696945
31: -10.9941368, 6.9561100, -10.9838638, 6.9532938, -14.6580505, 14.6592445
32: -24.9305458, -4.5432663, -24.9326134, -4.5749311, -13.2625046, 13.3368225
33: -69.3326263, -40.0832176, -69.3444519, -40.0917053, -16.6265717, 16.6884460
34: -53.7718697, -30.8915291, -53.7901001, -30.8986397, -14.1185684, 14.1766205
35: -47.8137207, -26.0630283, -47.8290710, -26.0597115, -13.0035782, 13.0243111
36: -42.8091354, -19.2733154, -42.8331413, -19.2730312, -15.0948029, 15.0997009
37: -86.7008514, -55.5359840, -86.6824188, -55.5379257, -18.9522705, 18.9353256
38: -52.9511452, -24.3141766, -52.9732361, -24.3186760, -18.3217316, 18.3923492
39: -76.5616608, -44.6164093, -76.5677643, -44.6191635, -16.0978317, 16.1032028
40: -67.2910461, -43.5135117, -67.2716064, -43.5163803, -14.3194618, 14.3330784
41: -55.4469833, -32.9431458, -55.4527206, -32.9496307, -16.6629295, 16.7050705
42: -29.4664116, -9.8930874, -29.4719772, -9.8998442, -17.2277718, 17.2744255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 886

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5235955, upper bound: 12.5638365
time: 10.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5675748
time: 14.27 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 26.97 seconds
IS_A1_A2_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4014289, upper bound: 12.5631753
IS_A1_A2_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4211745, upper bound: 12.5655894
IS_A1_A2_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4343372, upper bound: 12.5631753
IS_A1_A2_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4542779, upper bound: 12.5655894
IS_A1_A2_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4422144, upper bound: 12.5631753
IS_A1_A2_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4621553, upper bound: 12.5655894
IS_A1_A2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4928652, upper bound: 12.5230647
IS_A1_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4950464, upper bound: 12.5655891
IS_A1_A2_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4561023, upper bound: 12.5233774
IS_A1_A2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4591163, upper bound: 12.5667498
IS_A1_A2_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4900250, upper bound: 12.5233774
IS_A1_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4929624, upper bound: 12.5667498
IS_A1_A2_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4975057, upper bound: 12.5233774
IS_A1_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5004658, upper bound: 12.5667498
IS_A1_A2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5309294, upper bound: 12.5233774
IS_A1_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5338450, upper bound: 12.5667498
IS_A2_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4370200
IS_A2_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4619806
IS_A2_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4705595
IS_A2_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4965453
IS_A2_A1_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5621836, upper bound: 12.4138692
IS_A2_A1_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5664146, upper bound: 12.4572936
IS_A2_A1_B1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5220173, upper bound: 12.4882732
IS_A2_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4925248
IS_A2_A1_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5220173, upper bound: 12.4948017
IS_A2_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4990493
IS_A2_A1_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5220173, upper bound: 12.5299726
IS_A2_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5664143, upper bound: 12.5342492
IS_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4544338
IS_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5663393, upper bound: 12.4786982
IS_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4890084
IS_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5663393, upper bound: 12.5138163
IS_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5636971, upper bound: 12.4660420
IS_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5674506, upper bound: 12.5091083
IS_A2_A1_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5234525, upper bound: 12.5470845
IS_A2_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5674503, upper bound: 12.5508635
IS_A2_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5619689, upper bound: 12.4862491
IS_A2_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5652797, upper bound: 12.5124337
IS_A2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5623429, upper bound: 12.4642308
IS_A2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5665427, upper bound: 12.5092051
IS_A2_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5221770, upper bound: 12.5465501
IS_A2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5665424, upper bound: 12.5506992
IS_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5635137, upper bound: 12.5049642
IS_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5664667, upper bound: 12.5302199
IS_A2_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5235955, upper bound: 12.5221107
IS_A2_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5258383
IS_A2_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.4893752, upper bound: 12.5638365
IS_A2_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5323473, upper bound: 12.5675748
IS_A2_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5235955, upper bound: 12.5638365
IS_A2_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 26.97
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5675748

## BFS IS instance: IS_A1_A2_B2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -12.1122017, 3.6657629, -12.1180038, 3.6729159, -13.8566628, 13.8674011
1: -3.6592097, 7.3883190, -3.6638548, 7.3933382, -8.4567490, 8.4802742
2: -0.7339783, 13.4093838, -0.7390267, 13.4247599, -13.4162292, 13.4172325
3: -1.1084716, 11.2653551, -1.1194606, 11.2908611, -11.9796715, 11.9571304
4: -11.1052208, 5.4663720, -11.1101055, 5.4773207, -14.6356735, 14.6497421
5: 1.8559475, 17.7117157, 1.8526258, 17.7321930, -15.8702545, 15.8590899
6: -39.8838043, -18.2865677, -39.9155235, -18.2548904, -15.0664444, 15.1140327
7: -3.5466602, 12.2231731, -3.5635233, 12.2419262, -13.5255051, 13.5695457
8: -6.6947632, 8.5585022, -6.7011518, 8.5670872, -12.0824814, 12.0671864
9: -4.7484388, 11.7095718, -4.7715569, 11.7113047, -12.9669075, 12.9719887
10: 1.3939018, 25.7162552, 1.3466988, 25.7358551, -20.8305206, 20.8842010
11: -11.4591675, 4.2845159, -11.4829063, 4.2862759, -15.7454433, 15.7674217
12: -11.8572464, 9.8169804, -11.8852177, 9.8259773, -14.9550972, 14.9648438
13: -18.5337334, 6.7184496, -18.5490856, 6.7233753, -16.5582886, 16.6117363
14: 5.0860071, 36.3988876, 5.0143270, 36.4189529, -26.5952301, 26.6459503
15: -8.6710043, 9.2558241, -8.6826792, 9.2701941, -17.9411983, 17.9385033
16: -16.6911354, 2.5307643, -16.7152901, 2.5338700, -14.7461395, 14.7834702
17: 6.2830343, 30.6403828, 6.2390251, 30.6550713, -17.1466179, 17.1797180
18: -14.3833561, 5.1035752, -14.3896503, 5.1176825, -14.3840179, 14.3833313
19: -20.2545605, -4.3263502, -20.2686691, -4.3232942, -14.5079117, 14.5176315
20: -2.4014595, 11.2014322, -2.4136741, 11.2130251, -12.5878754, 12.5839119
21: -11.0415936, 3.2478166, -11.0621643, 3.2495008, -14.2910938, 14.3099804
22: -3.6739612, 13.0950832, -3.6890011, 13.1018648, -14.9124756, 14.8837433
23: -14.5610600, 0.3361740, -14.5722055, 0.3433321, -14.2995262, 14.2812729
24: -19.9201965, -5.1245637, -19.9272747, -5.1207919, -9.2335625, 9.2387314
25: -5.4158478, 10.8425350, -5.4359722, 10.8547287, -13.7596626, 13.7453461
26: -20.9609394, 1.1865437, -20.9921761, 1.2002237, -19.2836075, 19.2260208
27: -15.9914408, 2.1411276, -16.0069656, 2.1617150, -13.1990700, 13.1744995
28: -12.7774887, 4.6305885, -12.7911024, 4.6395521, -17.4170418, 17.4216919
29: -5.5638437, 11.8803043, -5.5820179, 11.8852234, -14.9237976, 14.8923378
30: -10.0129137, 6.1995482, -10.0337105, 6.2054081, -13.5132065, 13.5174637
31: -10.9571571, 6.9412622, -10.9684877, 6.9475307, -14.6260262, 14.6197853
32: -24.8773994, -4.6247740, -24.9103432, -4.5957561, -13.2190323, 13.2222862
33: -69.3006058, -40.1556702, -69.3120346, -40.1249886, -16.6042786, 16.6090698
34: -53.7433090, -30.9725342, -53.7615395, -30.9351883, -14.0887566, 14.0690422
35: -47.8116226, -26.0947838, -47.8183212, -26.0780029, -12.9723625, 12.9680939
36: -42.8052177, -19.3288155, -42.8202820, -19.2988892, -15.0632172, 15.0290070
37: -86.6627655, -55.5926056, -86.6719360, -55.5664558, -18.8765717, 18.8631439
38: -52.9201202, -24.3927479, -52.9441605, -24.3561878, -18.2852707, 18.2980042
39: -76.5462265, -44.6561432, -76.5552292, -44.6380844, -16.0601845, 16.0457611
40: -67.2305527, -43.5860596, -67.2480164, -43.5538330, -14.3171043, 14.2744408
41: -55.4048195, -33.0210648, -55.4264450, -32.9870987, -16.6509476, 16.6166954
42: -29.4596176, -9.9135523, -29.4647064, -9.8959122, -17.2418709, 17.2028580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 937

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.3780050, upper bound: 12.5626390
time: 6.95 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4009307, upper bound: 12.5626390
time: 7.82 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -12.1154785, 3.6706047, -12.1194839, 3.6755395, -13.8623009, 13.8718262
1: -3.6602728, 7.3890843, -3.6645312, 7.3956513, -8.4619255, 8.4837666
2: -0.7356747, 13.4203854, -0.7403638, 13.4306259, -13.4222031, 13.4285355
3: -1.1111763, 11.2902489, -1.1205591, 11.3040123, -11.9977531, 11.9756584
4: -11.1073589, 5.4739614, -11.1111937, 5.4812198, -14.6419907, 14.6581993
5: 1.8574533, 17.7305908, 1.8513656, 17.7424507, -15.8849974, 15.8792248
6: -39.8952789, -18.2499256, -39.9175377, -18.2354965, -15.0874596, 15.1314201
7: -3.5520287, 12.2389307, -3.5647743, 12.2503510, -13.5395432, 13.5766678
8: -6.6982899, 8.5620174, -6.7033572, 8.5690451, -12.0869598, 12.0731544
9: -4.7624769, 11.7043371, -4.7791433, 11.7120390, -12.9847488, 12.9738274
10: 1.3539419, 25.7219429, 1.3257217, 25.7365589, -20.8595886, 20.9070969
11: -11.4750328, 4.2857332, -11.4913025, 4.2868571, -15.7618904, 15.7770357
12: -11.8751211, 9.8207521, -11.8946991, 9.8273172, -14.9718857, 14.9785652
13: -18.5379658, 6.7177081, -18.5514984, 6.7249441, -16.5662804, 16.6129456
14: 5.0348711, 36.4038315, 4.9872379, 36.4198265, -26.6367950, 26.6687317
15: -8.6846714, 9.2562618, -8.6900635, 9.2737751, -17.9584465, 17.9463253
16: -16.7054424, 2.5304363, -16.7230549, 2.5345280, -14.7710381, 14.7931671
17: 6.2539082, 30.6467018, 6.2237520, 30.6564369, -17.1705017, 17.1935616
18: -14.3885126, 5.1090727, -14.3926878, 5.1204581, -14.3941956, 14.3933926
19: -20.2652645, -4.3281016, -20.2742348, -4.3232374, -14.5227280, 14.5241852
20: -2.4037108, 11.2108345, -2.4165652, 11.2181768, -12.5942459, 12.5966415
21: -11.0589542, 3.2484727, -11.0707932, 3.2499323, -14.3088865, 14.3192654
22: -3.6825230, 13.0948267, -3.6935134, 13.1032915, -14.9221230, 14.8872643
23: -14.5724049, 0.3374217, -14.5779095, 0.3441305, -14.3118439, 14.2883797
24: -19.9281769, -5.1267977, -19.9316082, -5.1200280, -9.2414017, 9.2493286
25: -5.4360695, 10.8474293, -5.4465208, 10.8554630, -13.7705688, 13.7585449
26: -20.9882965, 1.1906819, -21.0065937, 1.2020459, -19.3026352, 19.2431564
27: -15.9979305, 2.1569633, -16.0083332, 2.1701469, -13.2088432, 13.1779861
28: -12.7829971, 4.6319046, -12.7939157, 4.6404710, -17.4234676, 17.4258194
29: -5.5758195, 11.8805552, -5.5882854, 11.8858128, -14.9336472, 14.8918800
30: -10.0287037, 6.2033005, -10.0420628, 6.2064695, -13.5241852, 13.5286674
31: -10.9654121, 6.9443760, -10.9730778, 6.9492636, -14.6371155, 14.6288528
32: -24.8881798, -4.5900555, -24.9122429, -4.5773120, -13.2422066, 13.2369194
33: -69.3038177, -40.1334457, -69.3123703, -40.1131516, -16.6016998, 16.6263161
34: -53.7507095, -30.9413738, -53.7624817, -30.9188728, -14.1046486, 14.0874367
35: -47.8137817, -26.0815983, -47.8185196, -26.0711651, -12.9792595, 12.9798965
36: -42.8113976, -19.3032894, -42.8207664, -19.2854137, -15.0823555, 15.0412216
37: -86.6656342, -55.5819778, -86.6734772, -55.5608902, -18.8847656, 18.8846588
38: -52.9284592, -24.3554897, -52.9454041, -24.3366070, -18.3038788, 18.3169403
39: -76.5526199, -44.6385880, -76.5575409, -44.6288071, -16.0625534, 16.0702782
40: -67.2395935, -43.5576324, -67.2496490, -43.5387840, -14.3236618, 14.2911758
41: -55.4141159, -32.9900131, -55.4280510, -32.9707565, -16.6651535, 16.6312943
42: -29.4530964, -9.9001560, -29.4664307, -9.8886604, -17.2421951, 17.2209091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 586
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 937

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.3969995, upper bound: 12.5650764
time: 10.36 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4206940, upper bound: 12.5650764
time: 13.24 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -12.1197119, 3.6690745, -12.1191607, 3.6733866, -13.8671494, 13.8708916
1: -3.6600847, 7.3904591, -3.6625676, 7.3935466, -8.4578514, 8.4849396
2: -0.7401726, 13.4161587, -0.7410773, 13.4249840, -13.4245834, 13.4299583
3: -1.1261324, 11.2882519, -1.1280204, 11.2917480, -11.9958992, 11.9886894
4: -11.1088133, 5.4724874, -11.1102476, 5.4786510, -14.6371002, 14.6559906
5: 1.8399043, 17.7321835, 1.8444605, 17.7330799, -15.8825684, 15.8877230
6: -39.9049301, -18.2605457, -39.9251022, -18.2543221, -15.0848846, 15.1563835
7: -3.5761921, 12.2537098, -3.5770233, 12.2428770, -13.5503998, 13.6137657
8: -6.6995735, 8.5624838, -6.7006598, 8.5672684, -12.0847626, 12.0862694
9: -4.7577457, 11.7203341, -4.7718854, 11.7153931, -12.9834213, 12.9811554
10: 1.3758650, 25.7264748, 1.3454566, 25.7406635, -20.8535156, 20.8934784
11: -11.4786034, 4.3050313, -11.4888592, 4.2863498, -15.7649536, 15.7938900
12: -11.8619452, 9.8234711, -11.8872652, 9.8268185, -14.9719963, 14.9611549
13: -18.5433464, 6.7337675, -18.5535030, 6.7235513, -16.5644035, 16.6216660
14: 5.0579958, 36.4127541, 5.0009584, 36.4191360, -26.6218185, 26.6727524
15: -8.7048998, 9.2824469, -8.6829395, 9.2813282, -17.9862289, 17.9653854
16: -16.6975670, 2.5315852, -16.7137108, 2.5343137, -14.7562637, 14.7964478
17: 6.2606988, 30.6540394, 6.2279415, 30.6556759, -17.1662178, 17.2038383
18: -14.3995371, 5.1138630, -14.3901272, 5.1217022, -14.4037209, 14.3924847
19: -20.2656002, -4.3238583, -20.2696323, -4.3223624, -14.5139084, 14.5239754
20: -2.4160864, 11.2160263, -2.4193132, 11.2135592, -12.6012688, 12.6058159
21: -11.0490265, 3.2555971, -11.0629492, 3.2500153, -14.2990417, 14.3185463
22: -3.6869762, 13.1054764, -3.6890171, 13.1038628, -14.9373894, 14.8988037
23: -14.5742731, 0.3416016, -14.5735893, 0.3457627, -14.3043213, 14.2886581
24: -19.9356918, -5.1116610, -19.9277744, -5.1142969, -9.2581024, 9.2477150
25: -5.4294786, 10.8522491, -5.4363532, 10.8587589, -13.7797089, 13.7533798
26: -20.9873505, 1.2044246, -20.9929199, 1.2085562, -19.3102188, 19.2365112
27: -15.9986248, 2.1449676, -16.0070343, 2.1629841, -13.1998749, 13.1798553
28: -12.7860470, 4.6329384, -12.7922583, 4.6400013, -17.4260483, 17.4251976
29: -5.5685930, 11.8860350, -5.5830383, 11.8860693, -14.9388008, 14.9014740
30: -10.0240507, 6.2114964, -10.0372715, 6.2060261, -13.5231247, 13.5342331
31: -10.9663563, 6.9445806, -10.9695435, 6.9477468, -14.6318130, 14.6322670
32: -24.8976078, -4.6015010, -24.9204178, -4.5953321, -13.2350044, 13.2588043
33: -69.3125000, -40.1427231, -69.3121338, -40.1211205, -16.6211777, 16.6196022
34: -53.7484207, -30.9666901, -53.7616577, -30.9336166, -14.0954819, 14.0756721
35: -47.8168640, -26.0890007, -47.8187103, -26.0765152, -12.9821167, 12.9726944
36: -42.8067856, -19.3243542, -42.8210068, -19.2984314, -15.0670013, 15.0345573
37: -86.6937103, -55.5624313, -86.6735687, -55.5518951, -18.9224396, 18.8879585
38: -52.9224052, -24.3891850, -52.9447212, -24.3558197, -18.2896385, 18.3031693
39: -76.5585785, -44.6473503, -76.5558167, -44.6354980, -16.0778275, 16.0531769
40: -67.2377472, -43.5849838, -67.2490387, -43.5525932, -14.3231926, 14.2839546
41: -55.4099960, -33.0192642, -55.4270325, -32.9865417, -16.6559448, 16.6205902
42: -29.4630280, -9.9099274, -29.4642048, -9.8954515, -17.2456093, 17.2099533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 937

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4338400, upper bound: 12.5370536
time: 30.35 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4338400, upper bound: 12.5626390
time: 16.83 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -12.1229687, 3.6739194, -12.1206560, 3.6760015, -13.8727875, 13.8753204
1: -3.6611438, 7.3912497, -3.6632285, 7.3958788, -8.4630280, 8.4884338
2: -0.7418728, 13.4271564, -0.7424176, 13.4308395, -13.4305534, 13.4412537
3: -1.1288027, 11.3131466, -1.1291089, 11.3048944, -12.0139847, 12.0072346
4: -11.1109524, 5.4800596, -11.1113253, 5.4825497, -14.6434555, 14.6644516
5: 1.8413944, 17.7510471, 1.8432150, 17.7433548, -15.8986511, 15.9078321
6: -39.9163704, -18.2239399, -39.9271126, -18.2349205, -15.1058884, 15.1737709
7: -3.5815215, 12.2694912, -3.5782666, 12.2513103, -13.5644379, 13.6208992
8: -6.7031255, 8.5659876, -6.7028747, 8.5692701, -12.0892410, 12.0922394
9: -4.7717791, 11.7150860, -4.7794399, 11.7161264, -13.0012817, 12.9830093
10: 1.3358564, 25.7321701, 1.3244858, 25.7413788, -20.8825989, 20.9163361
11: -11.4944820, 4.3062639, -11.4972715, 4.2869349, -15.7814169, 15.8035355
12: -11.8798885, 9.8272381, -11.8967457, 9.8281822, -14.9888229, 14.9748611
13: -18.5476036, 6.7330236, -18.5558968, 6.7251034, -16.5724335, 16.6228714
14: 5.0068712, 36.4176712, 4.9738226, 36.4200058, -26.6633530, 26.6955261
15: -8.7185631, 9.2828693, -8.6903238, 9.2849264, -18.0034904, 17.9731941
16: -16.7118912, 2.5312295, -16.7214851, 2.5349865, -14.7811623, 14.8061714
17: 6.2315950, 30.6603661, 6.2127042, 30.6570015, -17.1901093, 17.2176628
18: -14.4046783, 5.1193771, -14.3931236, 5.1244869, -14.4139271, 14.4025574
19: -20.2763100, -4.3255882, -20.2752037, -4.3223290, -14.5287323, 14.5305099
20: -2.4183564, 11.2254257, -2.4221966, 11.2187004, -12.6076317, 12.6185303
21: -11.0663815, 3.2562418, -11.0715618, 3.2504435, -14.3168249, 14.3278036
22: -3.6955500, 13.1052303, -3.6935506, 13.1052933, -14.9470215, 14.9022942
23: -14.5856628, 0.3428373, -14.5793180, 0.3465502, -14.3166275, 14.2957687
24: -19.9436741, -5.1138620, -19.9321156, -5.1135283, -9.2659225, 9.2583122
25: -5.4497128, 10.8571177, -5.4469004, 10.8594685, -13.7906380, 13.7665901
26: -21.0146751, 1.2085352, -21.0073433, 1.2103713, -19.3292885, 19.2536316
27: -16.0051212, 2.1608257, -16.0083828, 2.1713920, -13.2096596, 13.1833534
28: -12.7915773, 4.6342635, -12.7950563, 4.6409411, -17.4325180, 17.4293194
29: -5.5805550, 11.8862610, -5.5892816, 11.8866768, -14.9486084, 14.9010086
30: -10.0398464, 6.2152610, -10.0455971, 6.2070546, -13.5341148, 13.5454254
31: -10.9746265, 6.9476633, -10.9741182, 6.9494648, -14.6429291, 14.6413612
32: -24.9083824, -4.5667543, -24.9223022, -4.5769033, -13.2581673, 13.2734413
33: -69.3158264, -40.1203842, -69.3125000, -40.1092873, -16.6185989, 16.6368637
34: -53.7557869, -30.9355278, -53.7625732, -30.9173012, -14.1113586, 14.0940742
35: -47.8190231, -26.0758095, -47.8189659, -26.0696354, -12.9890175, 12.9845200
36: -42.8130112, -19.2988052, -42.8215179, -19.2850018, -15.0860977, 15.0467606
37: -86.6965637, -55.5518036, -86.6751099, -55.5462990, -18.9306335, 18.9094734
38: -52.9309082, -24.3519058, -52.9459572, -24.3361797, -18.3082695, 18.3221054
39: -76.5650101, -44.6297951, -76.5581512, -44.6262207, -16.0801964, 16.0777016
40: -67.2467499, -43.5565834, -67.2505798, -43.5375290, -14.3297386, 14.3007050
41: -55.4192734, -32.9881325, -55.4286652, -32.9701462, -16.6701660, 16.6351662
42: -29.4564629, -9.8965006, -29.4659252, -9.8881779, -17.2459564, 17.2279816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=96, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 586
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1521
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 937

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.4538093, upper bound: 12.5394383
time: 6.87 seconds

## Relational analysis of IS_A1_A2_B2_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4538093, upper bound: 12.5650763
time: 8.09 seconds

## BFS IS instance: IS_A1_A2_B2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -12.1108913, 3.6712124, -12.1164732, 3.6731467, -13.8566170, 13.8723183
1: -3.6548042, 7.3919182, -3.6605875, 7.3936367, -8.4589119, 8.4918652
2: -0.7328971, 13.4113750, -0.7377474, 13.4249811, -13.4168434, 13.4192123
3: -1.1085573, 11.2629986, -1.1194288, 11.2877378, -11.9856491, 11.9589367
4: -11.1003132, 5.4674168, -11.1063795, 5.4775705, -14.6334991, 14.6497345
5: 1.8555269, 17.7116528, 1.8526411, 17.7310944, -15.8717804, 15.8590117
6: -39.8910713, -18.2930260, -39.9156685, -18.2597427, -15.0707321, 15.1135330
7: -3.5481544, 12.2263899, -3.5629478, 12.2419682, -13.5263672, 13.5790749
8: -6.6943064, 8.5595684, -6.6994038, 8.5672426, -12.0833054, 12.0688534
9: -4.7562437, 11.7298441, -4.7735748, 11.7112579, -12.9713173, 12.9860115
10: 1.3798933, 25.7354641, 1.3424358, 25.7357273, -20.8421860, 20.9039459
11: -11.4665155, 4.2904592, -11.4845781, 4.2856598, -15.7521753, 15.7750378
12: -11.8629246, 9.8219948, -11.8854122, 9.8256798, -14.9600716, 14.9687347
13: -18.5363846, 6.7290258, -18.5494823, 6.7232704, -16.5602264, 16.6156502
14: 5.0676632, 36.4369469, 5.0092163, 36.4189911, -26.6105347, 26.6815414
15: -8.6682587, 9.2601318, -8.6799507, 9.2706871, -17.9389458, 17.9400826
16: -16.7016449, 2.5436411, -16.7169304, 2.5339501, -14.7533798, 14.8023262
17: 6.2746186, 30.6470146, 6.2381268, 30.6550980, -17.1603165, 17.1776543
18: -14.3796501, 5.1052380, -14.3858700, 5.1179714, -14.3869324, 14.3857117
19: -20.2561569, -4.3251987, -20.2673950, -4.3230553, -14.5135345, 14.5173607
20: -2.4060061, 11.2003651, -2.4138689, 11.2118759, -12.5892563, 12.5836296
21: -11.0436144, 3.2484632, -11.0604038, 3.2494209, -14.2930355, 14.3088665
22: -3.6804881, 13.0992851, -3.6888659, 13.1023684, -14.9246101, 14.8871956
23: -14.5655441, 0.3374486, -14.5725956, 0.3433533, -14.3020248, 14.2833443
24: -19.9188805, -5.1242142, -19.9257565, -5.1208224, -9.2351074, 9.2401505
25: -5.4181232, 10.8440046, -5.4355879, 10.8547773, -13.7670403, 13.7447586
26: -20.9599152, 1.1886659, -20.9889946, 1.2003334, -19.2873802, 19.2268677
27: -16.0013161, 2.1446135, -16.0068378, 2.1627574, -13.2005692, 13.1783333
28: -12.7858114, 4.6339240, -12.7912779, 4.6403999, -17.4262123, 17.4252014
29: -5.5661302, 11.8819370, -5.5820875, 11.8848944, -14.9311218, 14.8932190
30: -10.0181122, 6.2059960, -10.0346203, 6.2051687, -13.5178719, 13.5254898
31: -10.9648523, 6.9433346, -10.9683018, 6.9481096, -14.6326561, 14.6212654
32: -24.8873405, -4.6335902, -24.9104290, -4.6022644, -13.2289467, 13.2220688
33: -69.3322296, -40.1423874, -69.3117294, -40.1205673, -16.6376877, 16.6177444
34: -53.7707634, -30.9615631, -53.7615776, -30.9320717, -14.1153030, 14.0776062
35: -47.8291473, -26.0872517, -47.8181915, -26.0761871, -12.9919243, 12.9750481
36: -42.8256683, -19.3226471, -42.8202133, -19.2974529, -15.0797424, 15.0328484
37: -86.6734238, -55.5871429, -86.6716843, -55.5647736, -18.8838387, 18.8674393
38: -52.9466324, -24.3850021, -52.9441376, -24.3542595, -18.3005676, 18.3027725
39: -76.5626984, -44.6493225, -76.5549927, -44.6357346, -16.0763779, 16.0511131
40: -67.2515869, -43.5789948, -67.2478638, -43.5509148, -14.3155556, 14.2789097
41: -55.4286156, -33.0128479, -55.4264946, -32.9845581, -16.6629944, 16.6197777
42: -29.4658489, -9.9256802, -29.4649353, -9.9040604, -17.2458916, 17.2008362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=88, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 954
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 921
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 921
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 950
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 1285
type: A, layer: 1, pos: 1285
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1521
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 945

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 937

## Relational analysis of IS_A1_A2_B2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4192551, upper bound: 12.5626390
time: 14.83 seconds

## Relational analysis of IS_A1_A2_B2_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_A1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.4417184, upper bound: 12.5626390
time: 24.10 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 41.25 seconds
IS_A1_A2_B2_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.3780050, upper bound: 12.5626390
IS_A1_A2_B2_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4009307, upper bound: 12.5626390
IS_A1_A2_B2_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.3969995, upper bound: 12.5650764
IS_A1_A2_B2_A1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4206940, upper bound: 12.5650764
IS_A1_A2_B2_A1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4338400, upper bound: 12.5370536
IS_A1_A2_B2_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4338400, upper bound: 12.5626390
IS_A1_A2_B2_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4538093, upper bound: 12.5394383
IS_A1_A2_B2_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4538093, upper bound: 12.5650763
IS_A1_A2_B2_A1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4192551, upper bound: 12.5626390
IS_A1_A2_B2_A1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 41.25
Output dim: 14, lower bound: -12.4417184, upper bound: 12.5626390
IS_A1_A2_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.4621553, upper bound: 12.5655894
IS_A1_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.4950464, upper bound: 12.5655891
IS_A1_A2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.4591163, upper bound: 12.5667498
IS_A1_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.4929624, upper bound: 12.5667498
IS_A1_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5004658, upper bound: 12.5667498
IS_A1_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5338450, upper bound: 12.5667498
IS_A2_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4370200
IS_A2_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4619806
IS_A2_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5618068, upper bound: 12.4705595
IS_A2_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5651443, upper bound: 12.4965453
IS_A2_A1_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5621836, upper bound: 12.4138692
IS_A2_A1_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5664146, upper bound: 12.4572936
IS_A2_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4925248
IS_A2_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5664143, upper bound: 12.4990493
IS_A2_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5664143, upper bound: 12.5342492
IS_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4544338
IS_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5663393, upper bound: 12.4786982
IS_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5633718, upper bound: 12.4890084
IS_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5663393, upper bound: 12.5138163
IS_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5636971, upper bound: 12.4660420
IS_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5674506, upper bound: 12.5091083
IS_A2_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5674503, upper bound: 12.5508635
IS_A2_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5619689, upper bound: 12.4862491
IS_A2_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5652797, upper bound: 12.5124337
IS_A2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5623429, upper bound: 12.4642308
IS_A2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5665427, upper bound: 12.5092051
IS_A2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5665424, upper bound: 12.5506992
IS_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5635137, upper bound: 12.5049642
IS_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5664667, upper bound: 12.5302199
IS_A2_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5258383
IS_A2_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.4893752, upper bound: 12.5638365
IS_A2_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5323473, upper bound: 12.5675748
IS_A2_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5235955, upper bound: 12.5638365
IS_A2_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 41.25
Output dim: 14, lower bound: -12.5675750, upper bound: 12.5675748

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 23.43 + 1798.26 = 1821.68 seconds
