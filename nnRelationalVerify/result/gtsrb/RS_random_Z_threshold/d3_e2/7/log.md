## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 45.0013539996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3955650, 44.3955612)
1: (-13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096)
2: (-14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4909210, 35.4909248)
3: (-12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700)
4: (-21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189)
5: (-12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787)
6: (-50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5662613, 40.5662537)
7: (-16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798)
8: (-18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384)
9: (-16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6461983, 38.6462021)
10: (-24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8275909, 61.8275909)
11: (-24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975)
12: (-28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9375648, 46.9375610)
13: (-32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388)
14: (-23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1073761, 60.1073761)
15: (-18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818)
16: (-32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851)
17: (-17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2744904, 55.2744942)
18: (-25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630)
19: (-26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995)
20: (-21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145)
21: (-25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867)
22: (-22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133)
23: (-21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798)
24: (-32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783)
25: (-18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148)
26: (-29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081)
27: (-32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254776, 47.8254814)
28: (-21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660)
29: (-23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474)
30: (-29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9555359, 45.9555321)
31: (-26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585)
32: (-42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5713501, 47.5713501)
33: (-72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3344040, 61.3344116)
34: (-56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6207123, 43.6207085)
35: (-50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2550659, 48.2550697)
36: (-47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0323868, 52.0323792)
37: (-83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4293747, 58.4293747)
38: (-58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3017960, 61.3017883)
39: (-78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3207550, 65.3207550)
40: (-67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1823769, 41.1823730)
41: (-55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2860641, 42.2860641)
42: (-33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7043839, 37.7043839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.85 + 60.09 = 62.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -45.0464004, upper bound: 45.0464004

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 720

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1012

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0460592, upper bound: 45.0043604
time: 52.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0043604, upper bound: 45.0460592
time: 23.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 76.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 76.57
Output dim: 14, lower bound: -45.0460592, upper bound: 45.0043604
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 76.57
Output dim: 14, lower bound: -45.0043604, upper bound: 45.0460592

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3985405, 44.3992615
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4923782, 35.4927177
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5648384, 40.5651512
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6433678, 38.6425285
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8241272, 61.8232117
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9346771, 46.9338837
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1033325, 60.1020889
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2722473, 55.2716789
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8227386, 47.8233109
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9539719, 45.9541321
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5744705, 47.5741959
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3314438, 61.3321381
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6210938, 43.6211166
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2531662, 48.2532959
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0299911, 52.0305710
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4140091, 58.4176865
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2951660, 61.2966919
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3123627, 65.3143539
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1753769, 41.1770248
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2784119, 42.2802658
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7075119, 37.7073059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 951

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0335984, upper bound: 44.9917912
time: 30.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0335543, upper bound: 44.9918304
time: 48.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3992577, 44.3985443
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4927216, 35.4923782
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5651588, 40.5648460
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6425285, 38.6433640
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8232117, 61.8241310
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9338837, 46.9346771
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1020813, 60.1033401
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2716827, 55.2722473
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8233109, 47.8227425
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9541321, 45.9539680
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5741959, 47.5744705
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3321304, 61.3314438
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6211243, 43.6210938
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2532959, 48.2531662
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0305710, 52.0299988
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4176865, 58.4140091
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2966919, 61.2951660
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3143463, 65.3123627
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1770248, 41.1753807
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2802658, 42.2784119
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7073059, 37.7075119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1614

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9946539, upper bound: 45.0456168
time: 52.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0039162, upper bound: 45.0363312
time: 51.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 106.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 106.90
Output dim: 14, lower bound: -45.0335984, upper bound: 44.9917912
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 106.90
Output dim: 14, lower bound: -45.0335543, upper bound: 44.9918304
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 106.90
Output dim: 14, lower bound: -44.9946539, upper bound: 45.0456168
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 106.90
Output dim: 14, lower bound: -45.0039162, upper bound: 45.0363312

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3983650, 44.3992386
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4916763, 35.4928741
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5646248, 40.5647545
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6430321, 38.6426773
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8229446, 61.8222427
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9346619, 46.9338341
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1032333, 60.1031151
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2724915, 55.2711983
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8226166, 47.8235054
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9543457, 45.9533234
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5741348, 47.5735474
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3319473, 61.3288040
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6220093, 43.6170158
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2534485, 48.2528000
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0297546, 52.0303955
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4131317, 58.4144669
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2949982, 61.2973099
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3121109, 65.3121185
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1783295, 41.1756363
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2780075, 42.2788925
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7020226, 37.7041130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 965

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 739

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0327095, upper bound: 44.9381516
time: 30.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9740301, upper bound: 44.9907251
time: 48.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3985176, 44.3992615
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4923782, 35.4920158
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5644493, 40.5651512
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6433678, 38.6421967
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8231583, 61.8232117
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9346771, 46.9338684
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1033325, 60.1019936
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2717590, 55.2716789
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8227386, 47.8231850
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9531631, 45.9541321
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5738297, 47.5741959
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3281326, 61.3321381
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6169891, 43.6211166
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2526703, 48.2532959
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0298309, 52.0305710
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4107819, 58.4176865
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2951660, 61.2965164
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3101273, 65.3143539
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1739960, 41.1770248
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2770309, 42.2802658
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7043190, 37.7073059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1583

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0330544, upper bound: 44.9892770
time: 73.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0309857, upper bound: 44.9913326
time: 47.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4032288, 44.4013901
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4930916, 35.4926453
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5588150, 40.5573540
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6420746, 38.6423531
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8188477, 61.8207970
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9349327, 46.9359322
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1054764, 60.1058693
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2735710, 55.2748566
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8265038, 47.8254128
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9541855, 45.9541016
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5758209, 47.5766029
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3373566, 61.3383179
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6142998, 43.6173935
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2591248, 48.2605209
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0294189, 52.0284882
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4186096, 58.4149017
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2901611, 61.2866821
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3152847, 65.3132782
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1775360, 41.1759872
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2802773, 42.2784195
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7128410, 37.7119141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 888

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9879298, upper bound: 45.0394991
time: 29.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9884914, upper bound: 45.0389375
time: 26.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4021072, 44.4025116
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4929848, 35.4927483
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5576630, 40.5585098
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6415176, 38.6429062
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8198853, 61.8197632
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9351387, 46.9357224
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1046219, 60.1067276
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2742882, 55.2741470
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8259850, 47.8259315
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9542618, 45.9540176
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5763245, 47.5761032
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3390045, 61.3366623
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6174202, 43.6142731
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2606506, 48.2589951
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0290680, 52.0288391
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4185791, 58.4149361
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2882080, 61.2886276
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3152847, 65.3132782
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1776276, 41.1758919
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2802773, 42.2784157
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7117119, 37.7130432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 953

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 923

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9685672, upper bound: 45.0361577
time: 58.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0037426, upper bound: 45.0009635
time: 67.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 128.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -45.0327095, upper bound: 44.9381516
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 128.47
Output dim: 14, lower bound: -44.9740301, upper bound: 44.9907251
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -45.0330544, upper bound: 44.9892770
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -45.0309857, upper bound: 44.9913326
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -44.9879298, upper bound: 45.0394991
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -44.9884914, upper bound: 45.0389375
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -44.9685672, upper bound: 45.0361577
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 128.47
Output dim: 14, lower bound: -45.0037426, upper bound: 45.0009635

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3942413, 44.3946724
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4934731, 35.4947891
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5447159, 40.5490723
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6364822, 38.6341591
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8162231, 61.8137779
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9380035, 46.9374847
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0944328, 60.0915833
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2694550, 55.2676735
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8167572, 47.8182144
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9524536, 45.9518776
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5623817, 47.5624542
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3294754, 61.3269501
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6173782, 43.6135445
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2509155, 48.2508926
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0269775, 52.0282974
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4138947, 58.4155045
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2881012, 61.2920990
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3147736, 65.3157806
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1751328, 41.1733093
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2677231, 42.2711754
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7105789, 37.7107925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0205349, upper bound: 44.9364150
time: 87.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0309890, upper bound: 44.9258776
time: 29.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3979187, 44.3986740
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4923782, 35.4920349
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5641403, 40.5648880
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6432419, 38.6420288
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8230286, 61.8230362
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9343109, 46.9333649
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1031799, 60.1018028
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2717133, 55.2716446
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8223305, 47.8229027
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9530602, 45.9540672
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5721626, 47.5722809
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3282013, 61.3321686
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6169281, 43.6210251
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2530289, 48.2534981
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0298843, 52.0306091
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4112625, 58.4183540
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2953644, 61.2966614
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3101044, 65.3145676
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1738510, 41.1768990
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2770042, 42.2802467
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7039032, 37.7068176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1659

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0305935, upper bound: 44.9685920
time: 31.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0123470, upper bound: 44.9868142
time: 26.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3979340, 44.3986664
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4923935, 35.4920311
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5641861, 40.5648422
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6431961, 38.6420822
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8229828, 61.8230858
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9341583, 46.9335060
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1031494, 60.1018372
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2717285, 55.2716370
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8224602, 47.8227768
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9530830, 45.9540367
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5719109, 47.5725365
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3281555, 61.3322220
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6168976, 43.6210556
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2528763, 48.2536545
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0298843, 52.0306091
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4114456, 58.4181747
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2953339, 61.2966919
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3103333, 65.3143387
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1738739, 41.1768761
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2770119, 42.2802391
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7038422, 37.7068863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0085312, upper bound: 44.9906724
time: 26.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0303250, upper bound: 44.9689620
time: 31.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4036598, 44.4012947
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4932861, 35.4926071
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5600662, 40.5571594
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6419525, 38.6426582
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8188171, 61.8208580
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9347916, 46.9358978
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1051178, 60.1065903
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2733192, 55.2754593
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8266029, 47.8253670
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9540939, 45.9544678
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5762024, 47.5762863
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3373032, 61.3379440
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6138535, 43.6169739
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2587051, 48.2603111
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0297089, 52.0283203
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4198761, 58.4141541
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2909317, 61.2862320
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3157425, 65.3130417
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1788483, 41.1757431
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2817116, 42.2778854
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7135277, 37.7117462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1599

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9744800, upper bound: 45.0390235
time: 22.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9874543, upper bound: 45.0260883
time: 52.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4031334, 44.4013901
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4930496, 35.4926453
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5586166, 40.5573540
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6420746, 38.6422310
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8188477, 61.8207664
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9348907, 46.9359322
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1054764, 60.1055069
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2735710, 55.2745972
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8264503, 47.8254128
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9541855, 45.9540176
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5755157, 47.5766029
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3369980, 61.3383179
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6138840, 43.6173935
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2589111, 48.2605209
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0292511, 52.0284882
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4178619, 58.4149017
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2896805, 61.2866821
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3150406, 65.3132782
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1772919, 41.1759872
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2797432, 42.2784195
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7126732, 37.7119141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9228250, upper bound: 45.0385261
time: 50.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9880804, upper bound: 44.9675159
time: 53.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4039612, 44.4040451
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4855576, 35.4857712
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5582733, 40.5586929
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6417770, 38.6436005
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8340912, 61.8309364
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9284172, 46.9267654
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1018829, 60.1062393
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2679749, 55.2702446
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8240356, 47.8257256
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9535446, 45.9543648
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5735397, 47.5701447
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3138580, 61.3027573
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5708084, 43.5542145
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2347412, 48.2263260
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0287628, 52.0284424
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4048615, 58.3965874
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2954559, 61.2937469
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3046417, 65.2992020
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1722717, 41.1658401
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2782593, 42.2736969
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7123947, 37.7126617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 852

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9354967, upper bound: 45.0358721
time: 49.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9682946, upper bound: 45.0031698
time: 60.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4036407, 44.4043732
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4860077, 35.4853134
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5578461, 40.5591125
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6422195, 38.6431656
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8310547, 61.8339691
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9261742, 46.9289970
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1041336, 60.1039925
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2703857, 55.2678375
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8257751, 47.8239861
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9546127, 45.9532928
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5703659, 47.5733147
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3050995, 61.3115082
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5573654, 43.5676651
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2279739, 48.2330856
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0286713, 52.0285339
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4002380, 58.4011993
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2933197, 61.2958679
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3011932, 65.3026352
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1675720, 41.1705360
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2755585, 42.2763977
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7113190, 37.7137299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 934

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9936833, upper bound: 45.0004630
time: 52.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0032287, upper bound: 44.9908146
time: 65.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 120.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0205349, upper bound: 44.9364150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0309890, upper bound: 44.9258776
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0305935, upper bound: 44.9685920
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0123470, upper bound: 44.9868142
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0085312, upper bound: 44.9906724
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0303250, upper bound: 44.9689620
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9744800, upper bound: 45.0390235
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9874543, upper bound: 45.0260883
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9228250, upper bound: 45.0385261
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9880804, upper bound: 44.9675159
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9354967, upper bound: 45.0358721
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9682946, upper bound: 45.0031698
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 120.08
Output dim: 14, lower bound: -44.9936833, upper bound: 45.0004630
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.08
Output dim: 14, lower bound: -45.0032287, upper bound: 44.9908146

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3942108, 44.3946342
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4934120, 35.4948196
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5458221, 40.5452423
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6363220, 38.6344070
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8163834, 61.8128166
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9385300, 46.9357109
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0923882, 60.0921936
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2683029, 55.2678528
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8161011, 47.8184166
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9516830, 45.9511261
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5636902, 47.5586929
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3316956, 61.3253403
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6219025, 43.6120300
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2520599, 48.2494125
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0272217, 52.0276489
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4146042, 58.4137535
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2886658, 61.2908478
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3148575, 65.3155670
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1763687, 41.1707954
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2689247, 42.2677650
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7114754, 37.7081642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9831148, upper bound: 44.9354559
time: 62.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0193432, upper bound: 44.9019429
time: 58.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3942413, 44.3946419
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4934731, 35.4947281
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5408783, 40.5490723
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6364822, 38.6339989
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8152542, 61.8137779
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9362259, 46.9374847
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0944328, 60.0895424
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2694550, 55.2665253
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8167572, 47.8175507
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9524536, 45.9511032
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5586166, 47.5624542
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3278503, 61.3269501
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6158676, 43.6135445
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2494354, 48.2508926
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0263214, 52.0282974
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4121323, 58.4155045
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2868652, 61.2920990
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3145828, 65.3157806
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1726151, 41.1733093
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2643089, 42.2711754
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7079506, 37.7107925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1588

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0235361, upper bound: 44.9088884
time: 21.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0140218, upper bound: 44.9185497
time: 40.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3998032, 44.4010429
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4822578, 35.4770317
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5530701, 40.5620537
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6407547, 38.6386223
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8142166, 61.8252602
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9146500, 46.9203186
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0921097, 60.0815544
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2673874, 55.2658882
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8334656, 47.8224373
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9532585, 45.9541512
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5448608, 47.5551872
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1951904, 61.2286682
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4227982, 43.4658890
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1664734, 48.1873245
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0127640, 52.0173111
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2695999, 58.3078728
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2767258, 61.2814407
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2390671, 65.2606506
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0790520, 41.1076889
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1966858, 42.2176476
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6476402, 37.6651764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 968

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9597026, upper bound: 44.9678764
time: 54.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0298738, upper bound: 44.8977344
time: 31.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4002762, 44.4005699
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4773827, 35.4819107
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5613022, 40.5538216
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6398392, 38.6395378
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8252640, 61.8142204
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9212570, 46.9137115
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0829315, 60.0907288
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2659531, 55.2673149
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8218689, 47.8340302
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9531517, 45.9542580
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5550690, 47.5449753
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2247009, 61.1991425
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4617920, 43.4268913
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1868515, 48.1669464
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0165787, 52.0135040
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3008041, 58.2766724
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2801437, 61.2780228
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2561874, 65.2435226
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1046486, 41.0820999
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2144012, 42.1999359
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6622581, 37.6505508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9751057, upper bound: 44.9855959
time: 57.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0111564, upper bound: 44.9495268
time: 24.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3994522, 44.3998642
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4842491, 35.4845581
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5615692, 40.5618286
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6452751, 38.6445808
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8459473, 61.8399124
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9177094, 46.9131355
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1003189, 60.1021996
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2711754, 55.2715492
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8264008, 47.8347816
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9506836, 45.9521141
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5672073, 47.5668449
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3020706, 61.2944794
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5661507, 43.5542603
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2038498, 48.1904564
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0229187, 52.0216675
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3560638, 58.3482132
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2880707, 61.2865448
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2862015, 65.2812576
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1513748, 41.1488342
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2466545, 42.2423019
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7052612, 37.7080841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 951

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9570217, upper bound: 44.9521843
time: 24.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9700906, upper bound: 44.9391024
time: 24.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3991318, 44.4001846
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4849205, 35.4838905
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5611725, 40.5622330
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6456947, 38.6441650
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8397980, 61.8460503
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9137878, 46.9170494
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1035233, 60.0990067
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2716331, 55.2710915
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8344650, 47.8267174
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9511719, 45.9516296
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5662155, 47.5678291
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2904129, 61.3061371
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5500908, 43.5703125
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1896896, 48.2046127
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0209351, 52.0236664
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3415070, 58.3627815
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2852020, 61.2894287
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2772598, 65.2901993
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1458282, 41.1543770
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2390709, 42.2498817
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7050323, 37.7083130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9843413, upper bound: 44.9675982
time: 26.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0289622, upper bound: 44.9229770
time: 30.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4091187, 44.4043198
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4953842, 35.4943542
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5548935, 40.5505829
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6425743, 38.6426086
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8147125, 61.8175163
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9376335, 46.9392967
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1079941, 60.1088791
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2689781, 55.2728119
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8297348, 47.8279533
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9571381, 45.9582977
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5825882, 47.5841026
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3422852, 61.3442764
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6257019, 43.6322136
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2636871, 48.2664337
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0270844, 52.0248871
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4115829, 58.4032745
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2800751, 61.2718658
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3115158, 65.3079681
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1772995, 41.1738892
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2776527, 42.2727509
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7119827, 37.7082138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9618395, upper bound: 45.0264630
time: 24.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9618018, upper bound: 45.0264975
time: 30.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4066925, 44.4067459
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4950409, 35.4947090
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5534897, 40.5520248
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6418953, 38.6432800
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8154755, 61.8167686
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9381981, 46.9387360
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1073990, 60.1094666
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2706642, 55.2711220
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8291855, 47.8285027
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9579315, 45.9575081
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5839767, 47.5826645
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3436279, 61.3429184
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6290894, 43.6288261
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2648315, 48.2652893
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0262756, 52.0256958
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4089890, 58.4058495
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2765656, 61.2753677
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3106613, 65.3088303
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1769943, 41.1742020
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2765770, 42.2738266
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7099991, 37.7102356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1580

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9722199, upper bound: 45.0255501
time: 48.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9869154, upper bound: 45.0108483
time: 54.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4042053, 44.4024124
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4931259, 35.4927292
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5464096, 40.5422058
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6320114, 38.6345139
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8111420, 61.8147316
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9348030, 46.9358406
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0907745, 60.0943184
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2700729, 55.2716255
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8250160, 47.8239822
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9478569, 45.9458275
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5776901, 47.5789757
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3263550, 61.3241425
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6044312, 43.6048050
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2496185, 48.2481308
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0245361, 52.0222321
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3952560, 58.3847656
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2787857, 61.2721939
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3024445, 65.2965164
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1664352, 41.1610870
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2655716, 42.2595253
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7214890, 37.7229767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 941

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8988603, upper bound: 45.0364976
time: 44.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9208122, upper bound: 45.0146182
time: 55.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4009132, 44.4010162
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4832382, 35.4851074
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5659409, 40.5617294
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6375732, 38.6405678
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8358154, 61.8305130
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9302559, 46.9260216
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0934906, 60.1016579
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2690506, 55.2715378
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8235321, 47.8271599
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9558182, 45.9567337
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5725403, 47.5635262
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2863922, 61.2667694
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5345917, 43.5059280
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2213135, 48.2077217
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0271606, 52.0261078
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3925781, 58.3795166
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2963562, 61.2944031
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2977066, 65.2898483
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1581650, 41.1447601
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2674255, 42.2580490
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7101974, 37.7061920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 965

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9307811, upper bound: 45.0348650
time: 33.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9252569, upper bound: 45.0311150
time: 29.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4009285, 44.4009972
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4848862, 35.4834518
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5613022, 40.5663681
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6387482, 38.6393967
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8336639, 61.8326721
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9276695, 46.9286079
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0973053, 60.0978470
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2692642, 55.2713242
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254700, 47.8252182
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9559097, 45.9566498
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5669098, 47.5691528
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2778625, 61.2753143
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5225296, 43.5179939
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2161331, 48.2129059
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0264282, 52.0268555
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3877716, 58.3843307
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2960968, 61.2946472
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2952652, 65.2922745
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1511917, 41.1517372
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2626190, 42.2628593
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7059250, 37.7104645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9682622, upper bound: 45.0018977
time: 77.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9670095, upper bound: 45.0031381
time: 20.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4030304, 44.4039459
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4820023, 35.4817772
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5595245, 40.5603523
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6422806, 38.6434135
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8356552, 61.8383369
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9237404, 46.9261055
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1039124, 60.1037102
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2708855, 55.2679329
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8269043, 47.8253098
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9542007, 45.9527245
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5667572, 47.5693817
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2900085, 61.2971878
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5459290, 43.5561676
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2205048, 48.2257385
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0287094, 52.0285797
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3951416, 58.3972168
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2955933, 61.2984238
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2982178, 65.3003464
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1606750, 41.1654015
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2715988, 42.2732010
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7118225, 37.7146683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 951

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 919

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8515069, upper bound: 44.9902320
time: 52.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0026305, upper bound: 44.8395737
time: 28.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 82.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9831148, upper bound: 44.9354559
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0193432, upper bound: 44.9019429
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0235361, upper bound: 44.9088884
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0140218, upper bound: 44.9185497
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9597026, upper bound: 44.9678764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0298738, upper bound: 44.8977344
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9751057, upper bound: 44.9855959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0111564, upper bound: 44.9495268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9570217, upper bound: 44.9521843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9700906, upper bound: 44.9391024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9843413, upper bound: 44.9675982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0289622, upper bound: 44.9229770
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9618395, upper bound: 45.0264630
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9618018, upper bound: 45.0264975
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9722199, upper bound: 45.0255501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9869154, upper bound: 45.0108483
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.8988603, upper bound: 45.0364976
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9208122, upper bound: 45.0146182
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9307811, upper bound: 45.0348650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9252569, upper bound: 45.0311150
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9682622, upper bound: 45.0018977
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.9670095, upper bound: 45.0031381
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 82.84
Output dim: 14, lower bound: -44.8515069, upper bound: 44.9902320
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 82.84
Output dim: 14, lower bound: -45.0026305, upper bound: 44.8395737

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3886795, 44.3883095
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4891586, 35.4899940
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5169754, 40.5223465
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6253662, 38.6205025
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8074188, 61.8014221
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9401932, 46.9380951
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0788803, 60.0750160
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2700806, 55.2697411
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8162231, 47.8185501
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9316063, 45.9351425
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5652847, 47.5603600
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3140717, 61.3119965
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6030426, 43.5972137
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2361450, 48.2371483
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0197983, 52.0219727
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3888092, 58.3942947
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2733002, 61.2790146
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2989349, 65.3037415
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1779022, 41.1761208
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2522850, 42.2550049
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7251511, 37.7186852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 967

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0104943, upper bound: 44.9018267
time: 24.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0192269, upper bound: 44.8931145
time: 28.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3908195, 44.3911057
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4959450, 35.4982262
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4962807, 40.5190735
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6288986, 38.6225128
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8208160, 61.8172073
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9346352, 46.9375725
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0859489, 60.0737381
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2661400, 55.2594223
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8051338, 47.8092079
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9581871, 45.9554482
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5580025, 47.5738831
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3369141, 61.3398514
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6227875, 43.6269493
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2488632, 48.2502556
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0290833, 52.0327988
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4128494, 58.4165382
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2950516, 61.3045807
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3044434, 65.3039856
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1798706, 41.1880760
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2540054, 42.2702026
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6838989, 37.6927795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 497

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0066847, upper bound: 44.9088849
time: 30.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0235321, upper bound: 44.8916936
time: 29.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3907127, 44.3912086
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4969826, 35.4971848
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5108681, 40.5044861
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6250076, 38.6264038
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8186951, 61.8193359
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9363060, 46.9359016
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0786247, 60.0810661
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2623558, 55.2632103
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8084145, 47.8059273
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9567986, 45.9568329
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5700569, 47.5618248
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3407593, 61.3360291
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6292725, 43.6204567
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2487869, 48.2503357
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0308228, 52.0310669
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4131851, 58.4162140
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2993240, 61.3003159
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3027649, 65.3056717
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1873779, 41.1805611
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2633286, 42.2608833
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6899338, 37.6867447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 884

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9544289, upper bound: 44.9183841
time: 32.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0138563, upper bound: 44.8588282
time: 28.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4015503, 44.4041138
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4816971, 35.4765015
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5235291, 40.5403519
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6436386, 38.6393433
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8128281, 61.8245010
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9139671, 46.9208832
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0728607, 60.0560608
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2591019, 55.2555580
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8356552, 47.8243027
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9535370, 45.9544106
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5371170, 47.5509529
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1925278, 61.2342377
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.3918381, 43.4451942
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1613617, 48.1861267
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0145493, 52.0212631
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2193832, 58.2681580
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2763214, 61.2853546
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2484436, 65.2744980
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0332031, 41.0728226
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1549988, 42.1862259
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6223068, 37.6462326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0209548, upper bound: 44.8530361
time: 31.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9852262, upper bound: 44.8889379
time: 30.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3947449, 44.3942451
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4731216, 35.4770813
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5324554, 40.5308609
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6287842, 38.6256332
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8164368, 61.8028259
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9229355, 46.9161263
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0693436, 60.0735474
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2677307, 55.2691956
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8220177, 47.8341827
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9330940, 45.9382629
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5566444, 47.5466347
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2070847, 61.1858139
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4429245, 43.4120789
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1709366, 48.1546860
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0091553, 52.0078201
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2749939, 58.2572098
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2648010, 61.2661591
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2402649, 65.2316895
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1063385, 41.0875435
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1977539, 42.1871834
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6762238, 37.6613655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1547

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 755

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0100837, upper bound: 44.9080454
time: 28.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9546559, upper bound: 44.9473202
time: 30.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3976059, 44.3991394
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4832458, 35.4820671
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5601120, 40.5614319
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6478691, 38.6456566
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8325195, 61.8397827
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9137421, 46.9173203
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0962334, 60.0896759
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2700310, 55.2691116
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8359070, 47.8268013
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9499931, 45.9505272
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5670166, 47.5688972
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2672729, 61.2871704
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5563583, 43.5817947
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1852722, 48.2034302
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0231705, 52.0267181
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3110504, 58.3378906
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2875214, 61.2929535
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2584305, 65.2745132
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1268768, 41.1391258
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2224388, 42.2367630
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7000809, 37.7043495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9855593, upper bound: 44.9224316
time: 51.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9866621, upper bound: 44.9213510
time: 35.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4089432, 44.4042969
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4946823, 35.4945183
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5546684, 40.5501862
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6422386, 38.6427536
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8135223, 61.8165283
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9376335, 46.9392586
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1078949, 60.1099052
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2692261, 55.2723198
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8296051, 47.8281517
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9575043, 45.9574814
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5822372, 47.5834503
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3427734, 61.3409500
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6265984, 43.6281013
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2639618, 48.2659378
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0268555, 52.0247192
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4107056, 58.4000626
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2798920, 61.2724686
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3112793, 65.3057251
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1802559, 41.1725044
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2772293, 42.2713661
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7064972, 37.7050247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1770

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9607149, upper bound: 45.0259611
time: 45.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9613560, upper bound: 45.0253091
time: 25.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4090958, 44.4043198
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4953842, 35.4936600
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5544853, 40.5505829
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6425743, 38.6422806
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8137360, 61.8175163
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9376335, 46.9392929
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1079941, 60.1087837
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2684860, 55.2728119
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8297348, 47.8278275
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9563293, 45.9582977
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5819321, 47.5841026
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3389587, 61.3442764
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6215935, 43.6322136
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2631836, 48.2664337
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0269165, 52.0248871
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4083557, 58.4032745
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2800751, 61.2716675
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3092804, 65.3079681
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1759148, 41.1738892
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2762680, 42.2727509
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7088013, 37.7082138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 496

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 765

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9530415, upper bound: 44.9818116
time: 51.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9170567, upper bound: 45.0176403
time: 51.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3984146, 44.3958054
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4957657, 35.4947281
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5533676, 40.5519142
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6394730, 38.6400795
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8166046, 61.8177948
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9388962, 46.9402504
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1076126, 60.1096725
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2659264, 55.2683525
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8306274, 47.8296738
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9496841, 45.9512291
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5838623, 47.5826263
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3439331, 61.3432465
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6315956, 43.6320992
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2674942, 48.2686996
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0246506, 52.0236053
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3983307, 58.3917465
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2722473, 61.2688293
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3064423, 65.3034592
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1704865, 41.1658592
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2692795, 42.2645607
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7040482, 37.7005997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1774

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9285879, upper bound: 45.0243106
time: 62.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9285879, upper bound: 44.9819462
time: 36.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3957443, 44.3984718
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4950562, 35.4954376
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5533829, 40.5518951
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6386948, 38.6408615
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8165131, 61.8179016
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9397049, 46.9394302
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1076126, 60.1096802
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2678947, 55.2663803
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8303604, 47.8299408
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9516525, 45.9492683
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5839386, 47.5825539
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3439636, 61.3432236
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6323662, 43.6313286
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2682343, 48.2679596
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0241928, 52.0240707
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3948822, 58.3951836
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2700348, 61.2710419
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3052826, 65.3046036
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1686478, 41.1676903
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2673111, 42.2665329
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7003708, 37.7042847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 876

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9852924, upper bound: 44.9927623
time: 86.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9688326, upper bound: 45.0092230
time: 31.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4023399, 44.3963242
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4944458, 35.4925613
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5446320, 40.5405121
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6316986, 38.6337776
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8112869, 61.8146706
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9335518, 46.9366913
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0896606, 60.0939560
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2655640, 55.2701569
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8252831, 47.8236122
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9433708, 45.9443741
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5773010, 47.5784187
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3259506, 61.3233719
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6043167, 43.6050148
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2495346, 48.2485390
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0241394, 52.0209579
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3919525, 58.3739624
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2772751, 61.2673492
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3009720, 65.2917633
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1649094, 41.1556892
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2636795, 42.2534752
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7191734, 37.7164955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 951

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8823213, upper bound: 44.9967130
time: 30.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8592476, upper bound: 45.0194690
time: 61.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3981133, 44.4005432
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4929657, 35.4940453
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5447083, 40.5404282
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6312790, 38.6342010
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8110886, 61.8148613
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9356575, 46.9345856
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0904236, 60.0931969
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2686157, 55.2671013
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8246574, 47.8242531
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9463921, 45.9413528
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5771332, 47.5785789
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3255844, 61.3237686
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6046371, 43.6046867
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2500229, 48.2480545
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0232544, 52.0218353
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3844604, 58.3814507
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2739487, 61.2706757
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2976913, 65.2950363
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1610413, 41.1595573
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2595139, 42.2576447
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7150078, 37.7206612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8787115, upper bound: 45.0142838
time: 44.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9204777, upper bound: 44.9725406
time: 46.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4001923, 44.3997688
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4832230, 35.4850998
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5657120, 40.5616608
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6367035, 38.6398315
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8345337, 61.8271790
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9295921, 46.9250908
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0930481, 60.1015472
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2683487, 55.2713127
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8205872, 47.8259354
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9540939, 45.9557152
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5720367, 47.5623131
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2848511, 61.2632675
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5366287, 43.5050850
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2220840, 48.2063255
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0272675, 52.0260468
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3900604, 58.3743248
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2964020, 61.2943878
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2964172, 65.2871780
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1571884, 41.1426735
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2660904, 42.2552223
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7094879, 37.7044067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7996586, upper bound: 45.0342081
time: 57.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9301170, upper bound: 44.9038528
time: 22.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3996582, 44.4002953
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4832230, 35.4850998
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5658722, 40.5615005
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6368332, 38.6397018
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8324890, 61.8292198
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9293251, 46.9253540
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0933762, 60.1012154
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2688293, 55.2708359
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8222961, 47.8242302
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9548035, 45.9550056
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5713348, 47.5630226
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2828979, 61.2652206
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5337448, 43.5079613
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2199173, 48.2084885
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0270996, 52.0262070
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3873901, 58.3769913
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2963409, 61.2944565
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2950287, 65.2885590
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1560822, 41.1437798
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2645950, 42.2567139
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7084122, 37.7054749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 936

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8801940, upper bound: 45.0306837
time: 29.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9340537, upper bound: 44.9768775
time: 48.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4002228, 44.4005013
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4850922, 35.4837112
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5613441, 40.5664215
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6385536, 38.6391983
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8340378, 61.8331299
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9276810, 46.9286232
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0971375, 60.0976219
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2689590, 55.2707863
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254395, 47.8251953
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9557877, 45.9562149
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5664444, 47.5689545
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2774734, 61.2750397
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5224533, 43.5180244
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2162170, 48.2129822
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0263367, 52.0267715
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3865814, 58.3832626
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2960968, 61.2946777
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2952042, 65.2922134
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1501694, 41.1510048
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2616119, 42.2621727
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7039757, 37.7090111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9222210, upper bound: 45.0005330
time: 41.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9669024, upper bound: 44.9557814
time: 50.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4004364, 44.4002876
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4851456, 35.4836578
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5613518, 40.5664062
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6385460, 38.6392059
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8341141, 61.8330650
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9276886, 46.9286194
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0970764, 60.0976753
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2687302, 55.2710152
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254471, 47.8251877
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9554749, 45.9565315
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5667191, 47.5686913
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2775955, 61.2749405
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5225601, 43.5179214
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2162094, 48.2129898
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0263367, 52.0267639
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3867188, 58.3831406
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2961273, 61.2946548
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2952042, 65.2922134
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1504669, 41.1507111
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2619324, 42.2618523
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7044716, 37.7085152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 825

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 940

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9521297, upper bound: 45.0027771
time: 54.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9666188, upper bound: 44.9876514
time: 66.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3990326, 44.4008331
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4766006, 35.4778137
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5886803, 40.5892334
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6276245, 38.6235123
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8319244, 61.8332634
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9204788, 46.9217529
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0729561, 60.0624542
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2621689, 55.2564926
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8053551, 47.8086395
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9495316, 45.9471321
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.6027374, 47.6069984
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2518463, 61.2623138
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4951782, 43.4995041
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2270355, 48.2310257
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0291214, 52.0302429
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3815460, 58.3929024
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2937012, 61.2975235
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2889862, 65.2938309
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1785355, 41.2001686
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2985306, 42.3119926
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7532768, 37.7554703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 202

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 883

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9602820, upper bound: 44.8395170
time: 48.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0025738, upper bound: 44.7970826
time: 61.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 112.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0104943, upper bound: 44.9018267
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0192269, upper bound: 44.8931145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0066847, upper bound: 44.9088849
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0235321, upper bound: 44.8916936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9544289, upper bound: 44.9183841
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0138563, upper bound: 44.8588282
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0209548, upper bound: 44.8530361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9852262, upper bound: 44.8889379
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0100837, upper bound: 44.9080454
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9546559, upper bound: 44.9473202
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9855593, upper bound: 44.9224316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9866621, upper bound: 44.9213510
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9607149, upper bound: 45.0259611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9613560, upper bound: 45.0253091
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9530415, upper bound: 44.9818116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9170567, upper bound: 45.0176403
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9285879, upper bound: 45.0243106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9285879, upper bound: 44.9819462
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9852924, upper bound: 44.9927623
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9688326, upper bound: 45.0092230
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.8823213, upper bound: 44.9967130
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.8592476, upper bound: 45.0194690
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.8787115, upper bound: 45.0142838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9204777, upper bound: 44.9725406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.7996586, upper bound: 45.0342081
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9301170, upper bound: 44.9038528
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.8801940, upper bound: 45.0306837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9340537, upper bound: 44.9768775
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9222210, upper bound: 45.0005330
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9669024, upper bound: 44.9557814
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9521297, upper bound: 45.0027771
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9666188, upper bound: 44.9876514
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 112.02
Output dim: 14, lower bound: -44.9602820, upper bound: 44.8395170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 112.02
Output dim: 14, lower bound: -45.0025738, upper bound: 44.7970826

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3922806, 44.3908310
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4893570, 35.4906349
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5316277, 40.5341492
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6261215, 38.6208611
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8002472, 61.7913895
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9415436, 46.9372826
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0783920, 60.0753975
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2638855, 55.2649307
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8112068, 47.8151093
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9287796, 45.9340706
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5765800, 47.5676689
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3007050, 61.2944412
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6111984, 43.5990982
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2351303, 48.2342682
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0198288, 52.0212708
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3824844, 58.3847122
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2700729, 61.2744904
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2962952, 65.3002167
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1851120, 41.1785011
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2631645, 42.2615891
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7382126, 37.7239151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 852

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9705355, upper bound: 44.8730183
time: 28.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9819029, upper bound: 44.8616047
time: 45.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3911972, 44.3919106
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4897995, 35.4901886
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5287743, 40.5369911
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6257248, 38.6212578
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7973938, 61.7942505
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9393768, 46.9394455
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0792542, 60.0745316
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2652740, 55.2635422
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8127785, 47.8135414
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9305344, 45.9323120
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5725975, 47.5716629
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2965240, 61.2986221
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6049271, 43.6053696
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2332611, 48.2361374
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0190964, 52.0220108
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3792343, 58.3879700
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2687607, 61.2758026
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2954102, 65.3011017
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1802902, 41.1833229
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2588615, 42.2658844
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7303848, 37.7317429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1265

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0050438, upper bound: 44.8494298
time: 32.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9755568, upper bound: 44.8789990
time: 62.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3891525, 44.3902550
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4941406, 35.4969292
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4964981, 40.5192261
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6286392, 38.6223602
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8235092, 61.8192902
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9319496, 46.9341812
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0857430, 60.0735893
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2697678, 55.2622681
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8076172, 47.8126373
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9619408, 45.9584236
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5578384, 47.5736847
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3340149, 61.3359909
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6178589, 43.6203842
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2446518, 48.2446365
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0292358, 52.0329132
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4131927, 58.4167900
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2972565, 61.3067322
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3036652, 65.3029099
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1808014, 41.1888199
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2553177, 42.2711678
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6771469, 37.6871643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 856

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9992491, upper bound: 44.9014331
time: 48.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9994274, upper bound: 44.9012542
time: 71.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3899612, 44.3894386
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4946442, 35.4964180
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4964447, 40.5192795
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6287460, 38.6222534
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8228989, 61.8198891
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9312477, 46.9348869
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0857887, 60.0735397
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2689896, 55.2630577
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8085632, 47.8116837
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9611626, 45.9592094
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5577927, 47.5737343
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3330688, 61.3369522
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6162262, 43.6220245
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2432480, 48.2460442
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0291901, 52.0329514
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4131012, 58.4168739
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2972260, 61.3067780
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3033905, 65.3032074
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1806030, 41.1890182
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2549820, 42.2714920
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6782990, 37.6860161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1770

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0224240, upper bound: 44.8911757
time: 26.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0230582, upper bound: 44.8904899
time: 25.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3846359, 44.3838310
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4957657, 35.4956512
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4826050, 40.4855576
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6181412, 38.6170769
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8189926, 61.8195610
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9331474, 46.9352341
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0658798, 60.0635223
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2609634, 55.2614784
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8032913, 47.8021545
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9564247, 45.9566841
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5331535, 47.5341797
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3221588, 61.3231049
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5924568, 43.5928078
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2469940, 48.2482224
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0283432, 52.0292282
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4125977, 58.4157829
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2994766, 61.3005371
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3053436, 65.3094177
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1634254, 41.1646767
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2406654, 42.2438583
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6662369, 37.6700783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0097580, upper bound: 44.8585352
time: 50.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9189988, upper bound: 44.8547215
time: 53.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3997459, 44.3997574
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4810829, 35.4750328
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5206451, 40.5371246
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6432953, 38.6385460
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8127365, 61.8253860
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9127693, 46.9222488
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0724831, 60.0551643
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2590179, 55.2565346
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8373489, 47.8241730
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9525375, 45.9541512
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5332260, 47.5494995
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1900253, 61.2332001
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.3855286, 43.4425812
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1561890, 48.1839905
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0145416, 52.0212402
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2177582, 58.2679405
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2773285, 61.2851639
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2476654, 65.2743759
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0320435, 41.0726814
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1546745, 42.1861916
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6246948, 37.6458054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0106868, upper bound: 44.8527549
time: 52.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0206775, upper bound: 44.8426178
time: 27.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3900909, 44.3894043
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4737053, 35.4777145
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5160294, 40.5185089
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6288452
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6205139, 38.6146393
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8107986, 61.7954063
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9255867, 46.9187813
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0579376, 60.0583992
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2660141, 55.2671661
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8197327, 47.8319817
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9268799, 45.9335976
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5499496, 47.5391617
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1965332, 61.1778717
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4294662, 43.4019470
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1594391, 48.1460419
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0050507, 52.0047379
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2617416, 58.2472382
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2567139, 61.2600708
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2344742, 65.2273407
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0980225, 41.0813942
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1831627, 42.1762047
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6894302, 37.6710205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0065774, upper bound: 44.8736983
time: 52.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9759165, upper bound: 44.9045260
time: 21.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4088631, 44.4043579
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4899063, 35.4881744
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5596085, 40.5541115
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6420898, 38.6435051
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8117218, 61.8162270
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9348907, 46.9371338
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1057701, 60.1083641
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2644386, 55.2688713
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8346786, 47.8296013
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9481888, 45.9507141
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5871964, 47.5833549
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3494415, 61.3498001
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6192551, 43.6221619
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2508774, 48.2572784
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0246582, 52.0231247
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3951416, 58.3895302
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2760162, 61.2697601
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3236008, 65.3225021
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1783257, 41.1704254
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2792282, 42.2732239
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7127533, 37.7092552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1638

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9602135, upper bound: 45.0059687
time: 31.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9406603, upper bound: 45.0254604
time: 38.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4090004, 44.4042244
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4883423, 35.4897308
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5585861, 40.5551300
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6429901, 38.6426125
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8132172, 61.8147278
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9355164, 46.9365158
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1063576, 60.1077805
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2657661, 55.2675400
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8310471, 47.8332253
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9507370, 45.9481659
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5821381, 47.5884094
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3516388, 61.3476181
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6206665, 43.6207466
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2552948, 48.2528610
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0252380, 52.0225372
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4001770, 58.3844986
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2771912, 61.2685852
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3280563, 65.3180466
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1781807, 41.1705704
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2790909, 42.2733612
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7107239, 37.7112808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 526

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9551522, upper bound: 45.0191684
time: 23.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9551346, upper bound: 45.0191684
time: 28.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4047356, 44.4025116
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4939117, 35.4930458
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5512657, 40.5477066
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6417770, 38.6419525
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8146362, 61.8174553
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9389992, 46.9380951
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1070824, 60.1084061
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2694550, 55.2727242
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8295898, 47.8295059
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9560547, 45.9572945
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5804749, 47.5802155
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3379059, 61.3417740
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6189766, 43.6259041
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2610321, 48.2612534
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0269012, 52.0248795
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4081421, 58.4016266
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2799072, 61.2726746
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3091583, 65.3071976
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1757812, 41.1727180
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2762299, 42.2724266
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7083740, 37.7106018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8930336, upper bound: 45.0171127
time: 46.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9165570, upper bound: 44.9936705
time: 28.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3936386, 44.3922043
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4923401, 35.4921417
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5577736, 40.5575714
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6399536, 38.6404686
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8270645, 61.8256264
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9361382, 46.9358826
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1059189, 60.1082611
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2698364, 55.2712860
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8387642, 47.8405380
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9496422, 45.9509163
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5798492, 47.5772934
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3340149, 61.3301086
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6146545, 43.6096420
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2556305, 48.2529793
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0251617, 52.0242310
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3942871, 58.3869514
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2771759, 61.2753983
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3031464, 65.2994080
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1690521, 41.1643448
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2677460, 42.2627182
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7063866, 37.7037201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9280743, upper bound: 45.0179617
time: 62.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9222289, upper bound: 45.0237922
time: 85.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3955727, 44.3983154
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4918365, 35.4911613
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5622559, 40.5571671
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6397324, 38.6423950
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8167801, 61.8207207
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9366951, 46.9371033
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1063271, 60.1087227
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2635345, 55.2630959
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8242722, 47.8202324
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9468689, 45.9456558
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5884171, 47.5844498
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3509521, 61.3535614
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6147614, 43.6179504
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2492447, 48.2540245
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0233002, 52.0232239
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3803406, 58.3856163
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2708130, 61.2717056
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3133545, 65.3165512
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1687775, 41.1677628
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2667923, 42.2657394
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7021790, 37.7034607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1694

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 909

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9645108, upper bound: 45.0088633
time: 69.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9684715, upper bound: 45.0048643
time: 49.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4011765, 44.3961792
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4942398, 35.4925461
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5434952, 40.5396385
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6318207, 38.6337662
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8111572, 61.8144264
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9336510, 46.9358902
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0895615, 60.0939293
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2659378, 55.2701378
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8252792, 47.8238182
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9434471, 45.9443016
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5770721, 47.5775909
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3259125, 61.3222351
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6042709, 43.6042747
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2495346, 48.2484741
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0241089, 52.0211411
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3917847, 58.3737907
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2772293, 61.2678223
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3008652, 65.2911224
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1647949, 41.1567650
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2636070, 42.2539825
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7191048, 37.7176323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 956

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8232633, upper bound: 44.9834909
time: 52.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8232633, upper bound: 44.9834909
time: 55.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3966293, 44.3983536
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4924774, 35.4931374
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5256920, 40.5252800
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6324120, 38.6349869
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8127441, 61.8170433
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9351273, 46.9342003
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0904427, 60.0932274
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2692680, 55.2681351
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8318787, 47.8310966
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9450302, 45.9406929
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5759506, 47.5776062
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3095856, 61.3075180
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5881271, 43.5886497
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2372742, 48.2355309
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0231018, 52.0216446
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3676605, 58.3600769
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2716370, 61.2676086
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2885590, 65.2829895
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1572304, 41.1544266
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2580986, 42.2553024
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7165833, 37.7223396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 938

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8719939, upper bound: 45.0135521
time: 51.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8779789, upper bound: 45.0075550
time: 48.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3875809, 44.3841362
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4654007, 35.4622498
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5824394, 40.5771523
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6043396, 38.6154900
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7952652, 61.7969589
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.8988190, 46.9019547
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0459023, 60.0658264
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2435036, 55.2507668
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.7917061, 47.7861366
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9446640, 45.9473076
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5832977, 47.5726738
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2829895, 61.2609329
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5410004, 43.5098763
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2410889, 48.2299118
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0298233, 52.0270386
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4320984, 58.4037819
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2912750, 61.2856293
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2930374, 65.2803268
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.2047539, 41.1767693
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3113785, 42.2889519
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7141685, 37.7087898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1578

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7992552, upper bound: 45.0339402
time: 52.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7993569, upper bound: 45.0338224
time: 44.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3978996, 44.3956871
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4829597, 35.4840775
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5531311, 40.5543976
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6361389, 38.6388626
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8301926, 61.8269539
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9250908, 46.9230042
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0932465, 60.1011047
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2663269, 55.2695618
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8222580, 47.8240967
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9526749, 45.9553375
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5674896, 47.5608902
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2801285, 61.2623596
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5321045, 43.5065460
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2177048, 48.2063599
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0270081, 52.0260925
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3798218, 58.3634186
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2959900, 61.2939835
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2921371, 65.2823868
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1537323, 41.1399879
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2645416, 42.2548523
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7081871, 37.7053833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 480

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8760660, upper bound: 45.0306069
time: 27.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8801190, upper bound: 45.0265633
time: 28.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4004288, 44.4002838
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4850082, 35.4835777
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5591545, 40.5585632
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6366768, 38.6396599
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8339005, 61.8328018
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9270096, 46.9265594
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0944366, 60.0997086
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2686691, 55.2721558
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8253098, 47.8251228
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9553452, 45.9560509
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5650558, 47.5633545
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2759171, 61.2672958
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5201721, 43.5078239
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2148056, 48.2080460
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0257874, 52.0246735
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3866272, 58.3759308
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2957153, 61.2917633
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2943802, 65.2882767
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1482811, 41.1422119
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2595291, 42.2527199
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7036057, 37.7057304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1775

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9298801, upper bound: 45.0022240
time: 27.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9515766, upper bound: 44.9805293
time: 26.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3951416, 44.3959045
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4782753, 35.4791107
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5846062, 40.5879898
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6202965, 38.6143417
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8299026, 61.8305397
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9205894, 46.9218979
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0664482, 60.0529633
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2618103, 55.2559776
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.7981071, 47.8032837
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9500122, 45.9477463
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5876999, 47.5969925
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2428436, 61.2554016
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4787827, 43.4866180
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2312164, 48.2345886
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0293350, 52.0305328
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3814240, 58.3928032
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2977524, 61.3030319
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2919006, 65.2975235
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1809845, 41.2067070
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2887802, 42.3048325
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7434845, 37.7480087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1588

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1780

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9854076, upper bound: 44.7949858
time: 30.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9970021, upper bound: 44.7615215
time: 55.71 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 88.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9705355, upper bound: 44.8730183
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9819029, upper bound: 44.8616047
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0050438, upper bound: 44.8494298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9755568, upper bound: 44.8789990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9992491, upper bound: 44.9014331
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9994274, upper bound: 44.9012542
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0224240, upper bound: 44.8911757
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0230582, upper bound: 44.8904899
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0097580, upper bound: 44.8585352
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9189988, upper bound: 44.8547215
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0106868, upper bound: 44.8527549
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0206775, upper bound: 44.8426178
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -45.0065774, upper bound: 44.8736983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9759165, upper bound: 44.9045260
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9602135, upper bound: 45.0059687
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9406603, upper bound: 45.0254604
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9551522, upper bound: 45.0191684
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9551346, upper bound: 45.0191684
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8930336, upper bound: 45.0171127
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9165570, upper bound: 44.9936705
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9280743, upper bound: 45.0179617
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9222289, upper bound: 45.0237922
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9645108, upper bound: 45.0088633
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9684715, upper bound: 45.0048643
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8232633, upper bound: 44.9834909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8232633, upper bound: 44.9834909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8719939, upper bound: 45.0135521
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8779789, upper bound: 45.0075550
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.7992552, upper bound: 45.0339402
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.7993569, upper bound: 45.0338224
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8760660, upper bound: 45.0306069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.8801190, upper bound: 45.0265633
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9298801, upper bound: 45.0022240
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9515766, upper bound: 44.9805293
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9854076, upper bound: 44.7949858
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 88.45
Output dim: 14, lower bound: -44.9970021, upper bound: 44.7615215

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3900833, 44.3884315
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4893494, 35.4894257
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5259666, 40.5344543
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6254196, 38.6207314
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7973785, 61.7943153
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9392624, 46.9408264
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0791855, 60.0744057
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2651901, 55.2641830
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8130379, 47.8135223
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9295654, 45.9315414
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5705681, 47.5723000
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2962952, 61.2985916
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6030121, 43.6049461
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2325897, 48.2358932
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0193787, 52.0219269
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3805847, 58.3873444
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2702408, 61.2755737
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2960052, 65.3004379
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1790924, 41.1824646
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2581711, 42.2657242
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7313461, 37.7314911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8852166, upper bound: 44.8487158
time: 24.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0043330, upper bound: 44.7296234
time: 25.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3898888, 44.3894997
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4898682, 35.4900818
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5013809, 40.5231934
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6285934, 38.6230087
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8211136, 61.8195839
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9285240, 46.9327698
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0836639, 60.0719910
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2642021, 55.2596016
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8136292, 47.8131256
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9518356, 45.9524384
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5627594, 47.5736504
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3397369, 61.3457794
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6088562, 43.6160698
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2301788, 48.2373962
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0270081, 52.0313416
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3975677, 58.4063721
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2933197, 61.3040695
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3157043, 65.3199844
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1786575, 41.1869316
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2569733, 42.2733498
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6845398, 37.6902504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 936

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1459

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0221850, upper bound: 44.8873534
time: 45.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0185281, upper bound: 44.8909284
time: 24.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3900261, 44.3893661
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4883041, 35.4916458
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5003586, 40.5242119
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6294861, 38.6221161
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8226089, 61.8180809
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9291420, 46.9321518
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0842514, 60.0714073
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2655296, 55.2582703
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8100052, 47.8167496
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9543839, 45.9498863
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5577087, 47.5787086
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3419037, 61.3436050
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6102676, 43.6146545
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2345963, 48.2329788
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0276031, 52.0307541
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4026031, 58.4013367
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2945099, 61.3028793
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3201599, 65.3155212
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1785202, 41.1870766
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2568359, 42.2734871
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6825256, 37.6922760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9853706, upper bound: 44.8492936
time: 26.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9819163, upper bound: 44.8526572
time: 32.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3846664, 44.3838387
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4958878, 35.4957237
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4814224, 40.4837418
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6179733, 38.6178513
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8124847, 61.8133278
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9366379, 46.9395599
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0658989, 60.0646553
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2602005, 55.2607956
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8067970, 47.8046570
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9606857, 45.9607201
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5351562, 47.5356903
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3205109, 61.3206711
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5932236, 43.5934753
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2489853, 48.2504730
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0269318, 52.0276184
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4062729, 58.4074097
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2973785, 61.2986221
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3012238, 65.3040009
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1578712, 41.1568909
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2382202, 42.2404022
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6666489, 37.6703033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 842

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 952

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9518061, upper bound: 44.8220811
time: 50.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9732688, upper bound: 44.8006311
time: 23.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4038544, 44.4029617
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4861565, 35.4821815
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4988098, 40.5106926
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6418686, 38.6358795
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7927094, 61.7999878
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.8987045, 46.9045830
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0677185, 60.0513611
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2556686, 55.2538795
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8158569, 47.8072586
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9527512, 45.9551468
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5106888, 47.5224113
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1607208, 61.1957779
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4264450, 43.4719353
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1392593, 48.1620407
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0108948, 52.0164185
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.1908722, 58.2347794
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2729034, 61.2790298
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2358475, 65.2591400
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -40.9990425, 41.0323486
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1324844, 42.1586342
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.5842896, 37.5952911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 186

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9741668, upper bound: 44.8523204
time: 33.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0104308, upper bound: 44.8220590
time: 54.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4029388, 44.4038620
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4882393, 35.4800949
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.4942245, 40.5152817
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6406250, 38.6371231
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7873383, 61.8053703
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.8951035, 46.9081879
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0686798, 60.0503960
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2563705, 55.2531738
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8204346, 47.8026810
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9535294, 45.9543610
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5061417, 47.5269547
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1526184, 61.2038803
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4148712, 43.4835129
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1342392, 48.1670609
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0097351, 52.0175934
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.1846085, 58.2410469
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2711945, 61.2807312
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2324448, 65.2625427
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -40.9917183, 41.0396767
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1271286, 42.1639977
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.5741806, 37.6054077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 965

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 497

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0038738, upper bound: 44.8426133
time: 48.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0206736, upper bound: 44.8256392
time: 24.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3848038, 44.3823128
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4731293, 35.4771271
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5082703, 40.5125885
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6328888, 39.6283188
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6186523, 38.6121788
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8108673, 61.7954826
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9291267, 46.9236069
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0556183, 60.0553169
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2667618, 55.2678871
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8201599, 47.8323288
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9267349, 45.9334183
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5443420, 47.5365753
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.1958923, 61.1773834
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.4225998, 43.3967781
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.1561966, 48.1433792
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0054703, 52.0051422
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.2696915, 58.2549591
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2594910, 61.2619858
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2405243, 65.2325363
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.0996284, 41.0841446
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.1834145, 42.1766357
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6924896, 37.6737137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9842566, upper bound: 44.8730377
time: 50.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0059172, upper bound: 44.8514333
time: 29.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4088326, 44.4043732
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4890976, 35.4872513
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5424271, 40.5451050
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6518517, 38.6513519
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8021393, 61.8052864
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9352112, 46.9375038
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1013947, 60.1009445
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2630692, 55.2672234
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254738, 47.8216362
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9567871, 45.9607735
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5781174, 47.5761948
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3347397, 61.3392487
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6034355, 43.6103172
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2549973, 48.2611465
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0193634, 52.0190201
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3757019, 58.3738708
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2657776, 61.2618408
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3132706, 65.3146286
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1439056, 41.1437302
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2597694, 42.2574844
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7115402, 37.7127914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 984

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9349412, upper bound: 44.9892764
time: 27.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9435236, upper bound: 44.9806464
time: 25.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4088783, 44.4043198
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4889832, 35.4873619
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5505981, 40.5369263
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6499443, 38.6532593
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8007660, 61.8066635
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9352570, 46.9374619
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0983505, 60.1039886
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2627945, 55.2675018
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8267097, 47.8204002
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9582443, 45.9593124
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5800400, 47.5742722
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3389053, 61.3350830
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6074028, 43.6063499
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2547455, 48.2613945
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0205536, 52.0178299
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3794708, 58.3701019
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2680817, 61.2595291
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3157120, 65.3121872
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1516266, 41.1360054
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2634773, 42.2537651
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7162857, 37.7080460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 783

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9334201, upper bound: 45.0251954
time: 35.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9403975, upper bound: 45.0182506
time: 86.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4079323, 44.4041100
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4861679, 35.4884911
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5567093, 40.5519676
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6421738, 38.6427002
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8104706, 61.8127441
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9344635, 46.9351349
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1054878, 60.1088943
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2650299, 55.2656746
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8296890, 47.8322411
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9499664, 45.9459229
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5805359, 47.5849953
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3476334, 61.3393021
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6153641, 43.6095543
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2539749, 48.2503738
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0249176, 52.0221176
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3973770, 58.3789291
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2766113, 61.2688141
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3260956, 65.3142929
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1778793, 41.1653671
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2771492, 42.2696304
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7032661, 37.7044640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 826

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9390352, upper bound: 45.0187899
time: 23.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9547758, upper bound: 45.0030741
time: 24.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.4088783, 44.4031563
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4871063, 35.4875565
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5554276, 40.5532417
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6430740, 38.6417961
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8112335, 61.8119850
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9341278, 46.9354744
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1074715, 60.1069069
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2639084, 55.2667961
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8300705, 47.8318596
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9485016, 45.9473915
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5787277, 47.5868111
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3432999, 61.3436203
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6094742, 43.6154404
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2528152, 48.2515335
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0248260, 52.0222168
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3945999, 58.3817024
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2774200, 61.2680054
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3242950, 65.3160934
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1729813, 41.1702652
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2753639, 42.2714157
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7039070, 37.7038231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 982

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9473150, upper bound: 45.0113658
time: 48.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9473150, upper bound: 45.0113658
time: 46.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3983917, 44.3977432
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4922409, 35.4915695
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5493546, 40.5431480
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6337051, 38.6360550
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8107910, 61.8146019
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9394875, 46.9386368
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1053085, 60.1092072
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2670059, 55.2692680
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8306160, 47.8293915
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9576187, 45.9566727
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5727386, 47.5705070
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3266678, 61.3272934
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6112900, 43.6140594
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2597046, 48.2600021
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0264587, 52.0243149
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4012909, 58.3930511
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2829132, 61.2760925
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3052597, 65.3021545
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1664009, 41.1587944
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2698631, 42.2641640
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7063828, 37.7079315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1306

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8823568, upper bound: 45.0062170
time: 77.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8823568, upper bound: 45.0062170
time: 79.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3930588, 44.3921776
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4922562, 35.4920502
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5523567, 40.5578766
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6401596, 38.6403389
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8269119, 61.8255501
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9345741, 46.9356232
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1059990, 60.1065788
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2699738, 55.2699623
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8383293, 47.8406067
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9500046, 45.9505653
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5749893, 47.5767021
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3312378, 61.3294983
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6131134, 43.6112671
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2556381, 48.2528915
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0247345, 52.0245590
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3929596, 58.3867073
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2762604, 61.2763138
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3029785, 65.2992783
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1640854, 41.1637726
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2642899, 42.2628517
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6994057, 37.7028275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1289

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9261121, upper bound: 45.0174597
time: 26.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9275724, upper bound: 45.0159689
time: 29.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3936157, 44.3916206
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4922485, 35.4920578
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5580788, 40.5521545
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6398239, 38.6406822
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8269730, 61.8254852
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9358864, 46.9343147
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1042366, 60.1083412
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2685089, 55.2714157
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8388329, 47.8401031
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9492950, 45.9512749
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5792542, 47.5724297
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3333893, 61.3273315
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6162796, 43.6081047
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2555466, 48.2529831
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0254669, 52.0238266
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3940430, 58.3856354
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2780914, 61.2744675
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3030090, 65.2992477
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1684799, 41.1593819
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2678680, 42.2592697
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7055016, 37.6967392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8247217, upper bound: 45.0232887
time: 80.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9087044, upper bound: 44.9263774
time: 59.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3894539, 44.3937263
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4879532, 35.4882469
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5603256, 40.5503006
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6327438, 38.6374702
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8181458, 61.8214951
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9312935, 46.9296722
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1178513, 60.1240730
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2626801, 55.2620964
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8244553, 47.8209991
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9420395, 45.9392319
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5772934, 47.5693550
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3463745, 61.3429260
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6161270, 43.6105995
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2463074, 48.2479744
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0210648, 52.0202255
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3712921, 58.3728790
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2694244, 61.2701950
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3092194, 65.3106766
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1942978, 41.1869621
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2663193, 42.2610245
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6955910, 37.6948776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9592796, upper bound: 45.0088412
time: 37.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9644916, upper bound: 45.0036048
time: 48.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3909798, 44.3922043
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4889145, 35.4872818
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5553818, 40.5552521
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6348038, 38.6354141
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8175659, 61.8220901
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9292564, 46.9317093
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1216812, 60.1202469
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2625427, 55.2622375
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8250427, 47.8204117
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9404449, 45.9408302
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5733185, 47.5733299
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3403168, 61.3489914
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6074142, 43.6193199
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2431946, 48.2510796
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0203171, 52.0209808
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3675995, 58.3765678
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2693024, 61.2703171
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3074799, 65.3124084
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1879807, 41.1932793
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2620850, 42.2652588
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.6936073, 37.6968765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1579

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9015028, upper bound: 45.0046053
time: 46.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9682124, upper bound: 44.9379239
time: 26.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3993225, 44.4005089
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4833183, 35.4842873
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5255966, 40.5309334
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6210022, 38.6216507
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8239212, 61.8209343
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9301834, 46.9272461
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0975342, 60.1008263
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2787971, 55.2781868
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8433228, 47.8503685
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9463654, 45.9425507
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5794830, 47.5815926
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3104630, 61.3047104
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5632782, 43.5569038
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2272873, 48.2171326
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0260925, 52.0240326
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3650513, 58.3566055
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2716675, 61.2667770
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2885513, 65.2802124
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1759758, 41.1720505
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2730637, 42.2674179
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7173042, 37.7228928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 824

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8680717, upper bound: 45.0116009
time: 55.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8699161, upper bound: 45.0098433
time: 60.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3987961, 44.4010315
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4836235, 35.4839897
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5313339, 40.5251923
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6190720, 38.6235809
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8166275, 61.8282166
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9281845, 46.9292526
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0980530, 60.1003151
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2793159, 55.2776642
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8511429, 47.8425446
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9468842, 45.9420319
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5799255, 47.5811462
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3067703, 61.3084183
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5563965, 43.5637932
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2188721, 48.2255402
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0254822, 52.0246429
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3641815, 58.3574677
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2708130, 61.2676392
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2857895, 65.2829742
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1748466, 41.1731758
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2701950, 42.2702751
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7171364, 37.7230606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 967

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1631

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8711459, upper bound: 45.0069621
time: 51.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8773856, upper bound: 45.0012714
time: 53.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3889847, 44.3856239
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4636765, 35.4605942
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5793457, 40.5742569
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6030693, 38.6143379
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7945328, 61.7952194
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.8992996, 46.9025612
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0456772, 60.0657120
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2421227, 55.2487221
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.7932053, 47.7873039
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9449348, 45.9474411
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5885506, 47.5774384
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2876205, 61.2643967
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5372696, 43.5043907
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2412720, 48.2303085
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0281677, 52.0256500
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4302521, 58.4015923
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2813263, 61.2780533
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2958908, 65.2829666
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1943550, 41.1640167
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3111496, 42.2886391
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7073288, 37.7005463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1547

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 867

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7640764, upper bound: 45.0335464
time: 30.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.7989500, upper bound: 44.9985000
time: 29.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3890610, 44.3855400
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4637451, 35.4605179
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5795441, 40.5740547
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6031837, 38.6142273
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.7935257, 61.7962341
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.8994217, 46.9024391
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0457993, 60.0655975
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2414513, 55.2493896
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.7928696, 47.7876320
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9448051, 45.9475670
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5880623, 47.5779152
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2864609, 61.2655716
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5355148, 43.5061455
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2414932, 48.2300949
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0284271, 52.0253906
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4299164, 58.4019318
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2837067, 61.2756805
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2956772, 65.2831802
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1919975, 41.1663742
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.3110580, 42.2887306
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7059174, 37.7019539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 480
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2032

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7992126, upper bound: 45.0275711
time: 24.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7931076, upper bound: 45.0336783
time: 49.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3978958, 44.3958855
1: -13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096
2: -14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4828568, 35.4841003
3: -12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700
4: -21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189
5: -12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787
6: -50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5530319, 40.5541420
7: -16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798
8: -18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384
9: -16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6361465, 38.6389046
10: -24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8306503, 61.8273048
11: -24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975
12: -28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9233513, 46.9210434
13: -32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388
14: -23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.0930595, 60.1009598
15: -18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818
16: -32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851
17: -17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2668762, 55.2699432
18: -25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630
19: -26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995
20: -21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145
21: -25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867
22: -22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133
23: -21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798
24: -32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783
25: -18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148
26: -29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081
27: -32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8218498, 47.8239212
28: -21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660
29: -23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474
30: -29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9533234, 45.9558220
31: -26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585
32: -42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5671463, 47.5604401
33: -72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.2793121, 61.2612991
34: -56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.5306625, 43.5046959
35: -50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2164536, 48.2047501
36: -47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0269318, 52.0259933
37: -83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.3799438, 58.3635101
38: -58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.2961121, 61.2940598
39: -78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.2920456, 65.2822723
40: -67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1537933, 41.1400757
41: -55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2645683, 42.2548714
42: -33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7071495, 37.7046127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 497
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 2016
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 496
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7958584, upper bound: 45.0302621
time: 64.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8757211, upper bound: 44.9465847
time: 26.88 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 94.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8852166, upper bound: 44.8487158
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0043330, upper bound: 44.7296234
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0221850, upper bound: 44.8873534
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0185281, upper bound: 44.8909284
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9853706, upper bound: 44.8492936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9819163, upper bound: 44.8526572
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9518061, upper bound: 44.8220811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9732688, upper bound: 44.8006311
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9741668, upper bound: 44.8523204
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0104308, upper bound: 44.8220590
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0038738, upper bound: 44.8426133
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0206736, upper bound: 44.8256392
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9842566, upper bound: 44.8730377
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -45.0059172, upper bound: 44.8514333
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9349412, upper bound: 44.9892764
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9435236, upper bound: 44.9806464
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9334201, upper bound: 45.0251954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9403975, upper bound: 45.0182506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9390352, upper bound: 45.0187899
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9547758, upper bound: 45.0030741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9473150, upper bound: 45.0113658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9473150, upper bound: 45.0113658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8823568, upper bound: 45.0062170
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8823568, upper bound: 45.0062170
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9261121, upper bound: 45.0174597
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9275724, upper bound: 45.0159689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8247217, upper bound: 45.0232887
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9087044, upper bound: 44.9263774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9592796, upper bound: 45.0088412
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9644916, upper bound: 45.0036048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9015028, upper bound: 45.0046053
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.9682124, upper bound: 44.9379239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8680717, upper bound: 45.0116009
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8699161, upper bound: 45.0098433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8711459, upper bound: 45.0069621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8773856, upper bound: 45.0012714
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.7640764, upper bound: 45.0335464
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.7989500, upper bound: 44.9985000
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.7992126, upper bound: 45.0275711
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.7931076, upper bound: 45.0336783
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.7958584, upper bound: 45.0302621
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 94.15
Output dim: 14, lower bound: -44.8757211, upper bound: 44.9465847
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 94.15
Output dim: 14, lower bound: -44.8801190, upper bound: 45.0265633
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 94.15
Output dim: 14, lower bound: -44.9298801, upper bound: 45.0022240

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 62.94 + 7191.86 = 7254.81 seconds
